"""Evaluation: retrieval ablations + end-to-end answer quality (synopsis §7.6 / objective 9).

Retrieval   Hit@1/3/5 and MRR@10 against labelled evidence (document + page [+ content type]),
            overall and per modality, for every configuration in the ablation grid.
Generation  answer accuracy (expected answer string present), correct abstention on
            unanswerable questions, citation accuracy (a cited source is real evidence),
            and latency.

usage: python -m eval.run_eval [--embed-models a,b] [--rerankers a,b] [--generate local,extractive]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmrag.config import Settings  # noqa: E402
from mmrag.models import get_reranker  # noqa: E402
from mmrag.pipeline import RAGPipeline  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

ABLATIONS = {
    "dense": dict(use_bm25=False, use_clip=False, use_rerank=False, modality_boost=0),
    "dense+bm25": dict(use_bm25=True, use_clip=False, use_rerank=False, modality_boost=0),
    "dense+bm25+clip": dict(use_bm25=True, use_clip=True, use_rerank=False, modality_boost=0),
    "hybrid+rerank": dict(use_bm25=True, use_clip=True, use_rerank=True, rerank_prior=0.0, modality_boost=0),
    "hybrid+rerank+prior0.3": dict(use_bm25=True, use_clip=True, use_rerank=True, rerank_prior=0.3, modality_boost=0),
    "full (+modality routing)": dict(use_bm25=True, use_clip=True, use_rerank=True, rerank_prior=0.0, modality_boost=0.05),
}


def matches(src: dict, ev: dict) -> bool:
    return (
        src["source"] == ev["source"]
        and int(src["page"]) == int(ev["page"])
        and (ev.get("type") in (None, src["type"]))
    )


def first_hit_rank(sources: list[dict], evidence: list[dict]) -> int | None:
    for r, s in enumerate(sources, 1):
        if any(matches(s, e) for e in evidence):
            return r
    return None


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().replace(",", "").replace("±", "+-")).strip()


def answer_correct(ans: str, expected: list[str]) -> bool:
    a = norm(ans)
    return any(norm(x) in a for x in expected)


def abstained(ans: str) -> bool:
    return "could not find" in ans.lower() or "not found in the provided" in ans.lower()


def retrieval_eval(p: RAGPipeline, qa: list[dict], k: int = 10, **ov) -> dict:
    rows, lat = [], []
    for item in qa:
        if not item["evidence"]:
            continue
        t = time.perf_counter()
        docs = p.searcher.search(item["question"], k=k, **ov)
        lat.append(time.perf_counter() - t)
        srcs = [{"source": d.metadata["source"], "page": d.metadata["page"], "type": d.metadata["type"]} for d in docs]
        rows.append({"q": item["question"], "modality": item["modality"], "rank": first_hit_rank(srcs, item["evidence"])})

    def agg(rs):
        n = len(rs) or 1
        return {
            "n": len(rs),
            "hit@1": sum(1 for r in rs if r["rank"] == 1) / n,
            "hit@3": sum(1 for r in rs if r["rank"] and r["rank"] <= 3) / n,
            "hit@5": sum(1 for r in rs if r["rank"] and r["rank"] <= 5) / n,
            "mrr@10": sum(1 / r["rank"] for r in rs if r["rank"]) / n,
        }

    mods = sorted({r["modality"].split("-")[0] for r in rows})
    return {
        "overall": agg(rows),
        "by_modality": {m: agg([r for r in rows if r["modality"].split("-")[0] == m]) for m in mods},
        "latency_ms": {"mean": 1000 * statistics.mean(lat), "p95": 1000 * sorted(lat)[int(0.95 * (len(lat) - 1))]},
        "misses": [r["q"] for r in rows if not r["rank"] or r["rank"] > 5],
    }


def generation_eval(p: RAGPipeline, qa: list[dict]) -> dict:
    rows = []
    for item in qa:
        if item["modality"].endswith("visual"):
            continue  # "show me X" requests have no textual answer to grade
        out = p.answer(item["question"])
        cited = [s for s in out["sources"] if s["n"] in out["cited"]]
        ans = out["answer"]
        row = {
            "question": item["question"],
            "modality": item["modality"],
            "answer": ans,
            "latency_s": out["timings"]["total_s"],
        }
        if item["modality"] == "unanswerable":
            row["correct"] = abstained(ans)
        else:
            row["correct"] = answer_correct(ans, item["answers"]) and not abstained(ans)
            row["citation_ok"] = any(any(matches(s, e) for e in item["evidence"]) for s in cited)
        rows.append(row)
        print(f"  [{'ok' if row['correct'] else 'XX'}] {item['question'][:70]} -> {ans[:110]!r}", flush=True)

    ans_rows = [r for r in rows if r["modality"] != "unanswerable"]
    una = [r for r in rows if r["modality"] == "unanswerable"]
    lat = sorted(r["latency_s"] for r in rows)
    mods = sorted({r["modality"].split("-")[0] for r in ans_rows})
    return {
        "answer_accuracy": sum(r["correct"] for r in ans_rows) / len(ans_rows),
        "by_modality": {
            m: sum(r["correct"] for r in ans_rows if r["modality"].split("-")[0] == m)
            / max(1, sum(1 for r in ans_rows if r["modality"].split("-")[0] == m))
            for m in mods
        },
        "citation_accuracy": sum(r["citation_ok"] for r in ans_rows) / len(ans_rows),
        "abstention_accuracy": sum(r["correct"] for r in una) / max(1, len(una)),
        "latency_s": {"p50": statistics.median(lat), "p95": lat[int(0.95 * (len(lat) - 1))]},
        "rows": rows,
    }


def build(embed_model: str, llm_backend: str = "extractive", **kw) -> RAGPipeline:
    s = Settings(data_dir=HERE / ".store", text_embed_model=embed_model, llm_backend=llm_backend, **kw)
    p = RAGPipeline(s)
    for pdf in sorted(DATA.glob("*.pdf")):
        info = p.ingest(pdf)
        if not info.get("skipped"):
            print(f"  ingested {pdf.name}: {info['chunks']} in {info['timings']['total_s']}s", flush=True)
    return p


def fmt_table(results: dict) -> str:
    lines = ["| configuration | Hit@1 | Hit@3 | Hit@5 | MRR@10 | text H@3 | table H@3 | image H@3 | latency ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        o, m = r["overall"], r["by_modality"]
        lines.append(
            f"| {name} | {o['hit@1']:.3f} | {o['hit@3']:.3f} | {o['hit@5']:.3f} | {o['mrr@10']:.3f} | "
            f"{m.get('text', {}).get('hit@3', 0):.3f} | {m.get('table', {}).get('hit@3', 0):.3f} | "
            f"{m.get('image', {}).get('hit@3', 0):.3f} | {r['latency_ms']['mean']:.0f} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-models", default="sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5")
    ap.add_argument("--rerankers", default="cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base")
    ap.add_argument("--generate", default="extractive,local", help="LLM backends for end-to-end eval ('' to skip)")
    ap.add_argument("--best-embed", default=None, help="embedding model for the generation run (default: best MRR)")
    args = ap.parse_args()

    qa = json.loads((DATA / "qa.json").read_text())
    RESULTS.mkdir(exist_ok=True)
    report = {"retrieval": {}, "generation": {}}
    md = ["# Evaluation report", "", f"{len(qa)} questions over {len(list(DATA.glob('*.pdf')))} PDFs "
          "(3 synthetic multimodal documents + the project synopsis). Evidence = correct document + page "
          "(+ content type where the fact lives only in a table/figure).", ""]

    best = (None, -1.0)
    for em in [m for m in args.embed_models.split(",") if m]:
        print(f"\n== embedding model: {em}", flush=True)
        p = build(em)
        res = {}
        for name, ov in ABLATIONS.items():
            res[name] = retrieval_eval(p, qa, **ov)
            print(f"  {name:28s} {res[name]['overall']}", flush=True)
        # reranker comparison on the full configuration
        default_rr = p.searcher.reranker
        for rr in [r for r in args.rerankers.split(",") if r]:
            p.searcher.reranker = get_reranker(rr, p.s.resolved_device)
            name = f"full, reranker={rr.split('/')[-1]}"
            res[name] = retrieval_eval(p, qa, **ABLATIONS["full (+modality routing)"])
            print(f"  {name:28s} {res[name]['overall']}", flush=True)
        p.searcher.reranker = default_rr
        report["retrieval"][em] = res
        md += [f"## Retrieval - text embedding `{em}`", "", fmt_table(res), ""]
        top = max(r["overall"]["mrr@10"] for r in res.values())
        if top > best[1]:
            best = (em, top)

    gen_embed = args.best_embed or best[0]
    for backend in [b for b in args.generate.split(",") if b]:
        print(f"\n== end-to-end with LLM backend '{backend}' (embedding {gen_embed})", flush=True)
        p = build(gen_embed, llm_backend=backend)
        g = generation_eval(p, qa)
        report["generation"][backend] = {**g, "llm": p.llm_desc, "embedding": gen_embed}
        md += [
            f"## End-to-end - {p.llm_desc}", "",
            "| metric | value |", "|---|---|",
            f"| answer accuracy (answerable) | {g['answer_accuracy']:.3f} |",
            *[f"| - {m} questions | {v:.3f} |" for m, v in g["by_modality"].items()],
            f"| citation accuracy | {g['citation_accuracy']:.3f} |",
            f"| abstention accuracy (unanswerable) | {g['abstention_accuracy']:.3f} |",
            f"| latency p50 / p95 (s) | {g['latency_s']['p50']:.2f} / {g['latency_s']['p95']:.2f} |",
            "",
            "<details><summary>answers</summary>", "",
            *[f"- {'✅' if r['correct'] else '❌'} **{r['question']}** — {r['answer'][:300]}" for r in g["rows"]],
            "", "</details>", "",
        ]

    (RESULTS / "results.json").write_text(json.dumps(report, indent=1, default=str))
    (RESULTS / "report.md").write_text("\n".join(md))
    print(f"\nwrote {RESULTS / 'report.md'}")


if __name__ == "__main__":
    main()
