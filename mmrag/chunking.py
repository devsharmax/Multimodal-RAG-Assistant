"""Turns parsed elements into retrieval chunks (LangChain `Document`s).

* Text: paragraphs of one page are packed into ~chunk_size chunks, preferring to break at
  section boundaries; over-long paragraphs are split with LangChain's recursive splitter.
  Chunks never cross pages, so every citation points at exactly one page.
* Tables: one chunk per table; large tables are split by rows with the header repeated.
* Figures: one chunk whose text is caption + generated description + text found in the figure.

Each chunk carries a short "contextual header" (document title > section) that is prepended
for embedding / BM25 only, which helps short chunks that do not repeat their topic.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ingestion import Element, ParsedDocument


def chunk_id(doc_id: str, page: int, kind: str, n: int) -> str:
    # "<doc_id>#..." lets Pinecone delete a whole document by id prefix.
    return f"{doc_id}#p{page}-{kind}{n}"


def figure_text(caption: str, description: str, ocr: str, native: str) -> str:
    parts = []
    if caption:
        parts.append(caption)
    if description:
        parts.append(f"Description: {description}")
    seen = native or ocr
    if seen:
        parts.append(f"Text in figure: {seen[:1500]}")
    return "\n".join(parts) or "Figure"


def _split_table(md: str, size: int) -> list[str]:
    if len(md) <= size * 2:
        return [md]
    lines = md.splitlines()
    header = lines[:2] if len(lines) > 2 and set(lines[1].replace("|", "").strip()) <= set("-: ") else lines[:1]
    body = lines[len(header) :]
    parts, cur = [], []
    for ln in body:
        if cur and sum(len(x) for x in cur) + len(ln) > size * 2:
            parts.append("\n".join(header + cur))
            cur = []
        cur.append(ln)
    if cur:
        parts.append("\n".join(header + cur))
    return parts


def linearize_table(rows: list[list[str]]) -> list[str]:
    """One "header: value; header: value" line per row - embeds and matches far better than
    Markdown pipes. A placeholder header row (e.g. a title cell + "Col2") is skipped."""
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return [" | ".join(r) for r in rows]
    is_placeholder = lambda r: sum(1 for c in r if not c or re.fullmatch(r"col\d+", c, re.I)) >= len(r) - 1
    title = ""
    if is_placeholder(rows[0]):
        title = next((c for c in rows[0] if c and not re.fullmatch(r"col\d+", c, re.I)), "")
        rows = rows[1:]
    header, body = rows[0], rows[1:]
    lines = [title] if title else []
    for r in body:
        parts = [f"{h}: {c}" if h else c for h, c in zip(header, r) if c]
        if parts:
            lines.append("; ".join(parts))
    return lines


def build_chunks(parsed: ParsedDocument, chunk_size: int = 900, chunk_overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
    )
    docs: list[Document] = []
    base = {"doc_id": parsed.doc_id, "source": parsed.filename, "title": parsed.title}

    def add(kind: str, page: int, n: int, content: str, section: str, **extra):
        meta = {**base, "id": chunk_id(parsed.doc_id, page, kind, n), "page": page, "type": kind, "section": section}
        meta.update({k: v for k, v in extra.items() if v not in (None, "")})
        docs.append(Document(page_content=content, metadata=meta, id=meta["id"]))

    pages = sorted({e.page for e in parsed.elements})
    for p in pages:
        els = [e for e in parsed.elements if e.page == p]
        # ---- text ------------------------------------------------------------------
        n = 0
        buf: list[str] = []
        buf_sections: list[str] = []

        def flush():
            nonlocal n, buf, buf_sections
            if buf:
                sec = buf_sections[0] if buf_sections else ""
                add("text", p, n, "\n".join(buf), sec)
                n += 1
            buf, buf_sections = [], []

        for e in (e for e in els if e.type == "text"):
            t = e.content.strip()
            if not t:
                continue
            if len(t) > chunk_size:
                flush()
                for piece in splitter.split_text(t):
                    add("text", p, n, piece, e.section)
                    n += 1
                continue
            new_section = buf_sections and e.section != buf_sections[-1]
            cur_len = sum(len(x) + 1 for x in buf)
            # Break at a section boundary once the chunk has some substance, or on size.
            if buf and (cur_len + len(t) > chunk_size or (new_section and cur_len > chunk_size * 0.4)):
                flush()
            buf.append(t)
            if not buf_sections or buf_sections[-1] != e.section:
                buf_sections.append(e.section)
        flush()

        # ---- tables ----------------------------------------------------------------
        for i, e in enumerate(x for x in els if x.type == "table"):
            lin = linearize_table(e.rows) if e.rows else []
            parts = _split_table(e.content, chunk_size)
            per = max(1, -(-len(lin) // len(parts))) if lin else 0
            for j, part in enumerate(parts):
                content = (e.caption + "\n" if e.caption else "") + part
                table_text = "\n".join(lin[j * per : (j + 1) * per]) if lin else ""
                add("table", p, i * 100 + j, content, e.section, table_text=table_text, caption=e.caption)

        # ---- figures (description is filled in later by the pipeline) ---------------
        for i, e in enumerate(x for x in els if x.type == "image"):
            add(
                "image",
                p,
                i,
                figure_text(e.caption, "", "", e.content),
                e.section,
                image_path=e.image_path,
                caption=e.caption,
                native_text=e.content[:1500],
                origin=e.origin,
            )
    return docs


def embed_text(doc: Document, contextual: bool = True) -> str:
    """Text actually fed to the embedder / BM25: optional contextual header + content."""
    m = doc.metadata
    body = doc.page_content
    if m.get("type") == "table" and m.get("table_text"):
        body = (m.get("caption", "") + "\n" + m["table_text"]).strip()
    if not contextual:
        return body
    header = " > ".join(x for x in (m.get("title", ""), m.get("section", "")) if x)
    label = {"table": "Table", "image": "Figure"}.get(m.get("type"), "")
    return f"{header}\n{label + ': ' if label else ''}{body}" if header else body


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was were what when "
    "where which who why will with does do did can could should would i you we they he she them our your their "
    "there these those than then so such into about over under".split()
)


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN_RE.findall(text.lower()):
        if t in STOPWORDS:
            continue
        toks.append(t)
        if any(c in t for c in ".-_"):  # also index the parts of compound tokens
            toks.extend(x for x in re.split(r"[.\-_]", t) if x and x not in STOPWORDS)
    return toks
