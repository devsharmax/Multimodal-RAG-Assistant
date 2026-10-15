"""Document Ingestion Module (synopsis §6.1).

Walks every page of a PDF with PyMuPDF and extracts four kinds of content, each tagged with
source / page / type metadata:

* text paragraphs, with their section heading (from font size / boldness) and with repeating
  headers/footers removed;
* tables (PyMuPDF table finder -> Markdown);
* embedded raster images (rendered at their on-page position, tiny icons dropped);
* vector figures (charts and diagrams drawn as PDF paths, found by clustering drawings),
  which plain image extraction misses entirely.

Scanned pages that have no text layer are OCR'd. Failures are isolated per page so one bad
page never aborts the whole document (synopsis §10.4).
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

log = logging.getLogger(__name__)

CAPTION_RE = re.compile(r"^\s*(fig(ure)?|table|chart|diagram|graph|plate|exhibit)\.?\s*\d+", re.I)
HEADING_NUM_RE = re.compile(r"^\s*(\d+(\.\d+)*|[IVX]+\.|[A-Z]\.)\s+\S")


@dataclass
class Element:
    type: str  # "text" | "table" | "image"
    page: int  # 1-based
    content: str
    bbox: tuple[float, float, float, float]
    section: str = ""
    image_path: str | None = None
    caption: str = ""
    origin: str = ""  # raster | vector | scan | native
    rows: list[list[str]] | None = None  # table cells, for row linearisation


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    title: str
    n_pages: int
    elements: list[Element] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _overlap_frac(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Fraction of `a`'s area that lies inside `b`."""
    inter = pymupdf.Rect(a) & b
    if inter.is_empty or a.get_area() == 0:
        return 0.0
    return inter.get_area() / a.get_area()


def _norm_margin(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _block_text(block: dict) -> tuple[str, float, bool]:
    """Return (text, max font size, all-bold) for a PyMuPDF text block."""
    lines, sizes, bold_chars, chars = [], [], 0, 0
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        t = "".join(s.get("text", "") for s in spans)
        if t.strip():
            lines.append(t.strip())
        for s in spans:
            n = len(s.get("text", "").strip())
            if not n:
                continue
            sizes.append(s.get("size", 0))
            chars += n
            if s.get("flags", 0) & 16 or "bold" in s.get("font", "").lower():
                bold_chars += n
    text = " ".join(lines)
    text = re.sub(r"-\s+(?=[a-z])", "", text) if text.count("- ") > 2 else text
    text = re.sub(r"\s+", " ", text).strip()
    return text, max(sizes) if sizes else 0.0, chars > 0 and bold_chars / chars > 0.8


class PDFParser:
    def __init__(self, images_dir: Path, render_dpi: int = 150, min_image_px: int = 80, ocr=None):
        self.images_dir = Path(images_dir)
        self.dpi = render_dpi
        self.min_px = min_image_px
        self.ocr = ocr  # callable(path) -> str, used for scanned pages

    # ------------------------------------------------------------------------------------
    def parse(self, pdf_path: str | Path, doc_id: str, filename: str | None = None) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        doc = pymupdf.open(pdf_path)
        filename = filename or pdf_path.name
        title = (doc.metadata or {}).get("title") or ""
        out = ParsedDocument(doc_id=doc_id, filename=filename, title=title, n_pages=doc.page_count)
        img_dir = self.images_dir / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)

        page_blocks = []
        for page in doc:
            try:
                page_blocks.append(page.get_text("dict", sort=True)["blocks"])
            except Exception as e:  # pragma: no cover - defensive
                page_blocks.append([])
                out.warnings.append(f"page {page.number + 1}: text extraction failed ({e})")

        margin_repeats = self._repeating_margins(doc, page_blocks)
        body_size = self._body_font_size(page_blocks)
        section = ""

        for page, blocks in zip(doc, page_blocks):
            try:
                section = self._parse_page(page, blocks, out, img_dir, margin_repeats, body_size, section)
            except Exception as e:
                log.exception("page %s failed", page.number + 1)
                out.warnings.append(f"page {page.number + 1}: {type(e).__name__}: {e}")

        if not out.title and page_blocks and page_blocks[0]:
            cands = [_block_text(b) for b in page_blocks[0] if b.get("type") == 0]
            cands = [(size, t) for t, size, _ in cands if 3 < len(t) < 200]
            if cands:
                out.title = max(cands, key=lambda c: c[0])[1]
        out.title = out.title or Path(filename).stem
        doc.close()
        return out

    # ------------------------------------------------------------------------------------
    @staticmethod
    def _repeating_margins(doc, page_blocks) -> set[str]:
        if doc.page_count < 2:
            return set()
        counts: Counter[str] = Counter()
        for page, blocks in zip(doc, page_blocks):
            h = page.rect.height
            seen = set()
            for b in blocks:
                if b.get("type") != 0:
                    continue
                y0, y1 = b["bbox"][1], b["bbox"][3]
                if y1 < 0.09 * h or y0 > 0.91 * h:
                    t, _, _ = _block_text(b)
                    if t:
                        seen.add(_norm_margin(t))
            counts.update(seen)
        thresh = max(2, int(0.4 * doc.page_count))
        return {t for t, c in counts.items() if c >= thresh}

    @staticmethod
    def _body_font_size(page_blocks) -> float:
        sizes: Counter[float] = Counter()
        for blocks in page_blocks:
            for b in blocks:
                for line in b.get("lines", []):
                    for s in line.get("spans", []):
                        sizes[round(s.get("size", 0), 1)] += len(s.get("text", ""))
        return sizes.most_common(1)[0][0] if sizes else 11.0

    # ------------------------------------------------------------------------------------
    def _parse_page(self, page, blocks, out: ParsedDocument, img_dir: Path, margin_repeats, body_size, section):
        pno = page.number + 1
        prect = page.rect
        page_area = prect.get_area()

        # ---- tables ---------------------------------------------------------------------
        tables: list[tuple[pymupdf.Rect, str, list]] = []
        try:
            for t in page.find_tables().tables:
                if t.row_count < 2 or t.col_count < 2:
                    continue
                md = t.to_markdown(clean=False).strip()
                rows = [[re.sub(r"\s+", " ", str(c or "")).strip() for c in row] for row in (t.extract() or [])]
                if sum(1 for row in rows for c in row if c) < 4:
                    continue
                # Boxes drawn side by side in a diagram look like a 1-2 row table whose header
                # is mostly empty / "ColN" placeholders: that is a figure, not a table.
                hdr = rows[0] if rows else []
                blank = sum(1 for c in hdr if not c or re.fullmatch(r"col\d+", c, re.I))
                if len(rows) <= 2 and hdr and blank >= 0.4 * len(hdr):
                    continue
                tables.append((pymupdf.Rect(t.bbox), md, rows))
        except Exception as e:
            out.warnings.append(f"page {pno}: table detection failed ({e})")

        # ---- text blocks ----------------------------------------------------------------
        text_blocks = []
        native_chars = 0
        for b in blocks:
            if b.get("type") != 0:
                continue
            t, size, bold = _block_text(b)
            if not t:
                continue
            native_chars += len(t)
            if _norm_margin(t) in margin_repeats or re.fullmatch(r"(page\s*)?\d{1,4}", t.strip(), re.I):
                continue
            text_blocks.append((pymupdf.Rect(b["bbox"]), t, size, bold))

        # headings on this page, so figures/tables get the section they sit in
        headings = [(r.y0, t[:120]) for r, t, size, bold in text_blocks if self._is_heading(t, size, bold, body_size)]

        def section_at(y: float) -> str:
            cur = section
            for hy, ht in headings:
                if hy <= y:
                    cur = ht
            return cur

        # ---- raster images --------------------------------------------------------------
        figures: list[tuple[pymupdf.Rect, str]] = []
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                rects = []
            for r in rects:
                r = pymupdf.Rect(r) & prect
                if r.is_empty:
                    continue
                if r.get_area() > 0.9 * page_area:
                    if native_chars < 50 and self.ocr is not None:
                        figures.append((r, "scan"))
                    continue
                figures.append((r, "raster"))

        # ---- vector figures (charts / diagrams drawn with paths) ------------------------
        figures += [(r, "vector") for r in self._vector_figures(page, tables, [f[0] for f in figures])]

        # ---- dedupe figures -------------------------------------------------------------
        kept: list[tuple[pymupdf.Rect, str]] = []
        for r, origin in sorted(figures, key=lambda f: -f[0].get_area()):
            if any(_overlap_frac(r, k[0]) > 0.6 for k in kept):
                continue
            kept.append((r, origin))

        # ---- emit figure elements -------------------------------------------------------
        fig_rects = []
        for i, (r, origin) in enumerate(sorted(kept, key=lambda f: (f[0].y0, f[0].x0))):
            w_px = r.width / 72 * self.dpi
            h_px = r.height / 72 * self.dpi
            if origin != "scan" and (w_px < self.min_px or h_px < self.min_px or r.get_area() < 0.01 * page_area):
                continue
            path = img_dir / f"p{pno}_{i}.png"
            page.get_pixmap(clip=r, dpi=self.dpi).save(path)
            if origin == "scan":
                text = self.ocr(str(path)) if self.ocr else ""
                if text.strip():
                    out.elements.append(Element("text", pno, text, tuple(r), section_at(r.y0), origin="scan"))
                continue
            native = page.get_text("text", clip=r).strip() if origin == "vector" else ""
            caption = self._find_caption(r, text_blocks)
            out.elements.append(
                Element("image", pno, re.sub(r"\s+", " ", native), tuple(r), section_at(r.y0), str(path), caption, origin)
            )
            fig_rects.append(r)

        for r, md, rows in tables:
            caption = self._find_caption(r, text_blocks)
            md = md.replace("<br>", " ")
            out.elements.append(
                Element("table", pno, md, tuple(r), section_at(r.y0), caption=caption, origin="native", rows=rows)
            )

        # ---- body text (excluding table cells and figure labels) ------------------------
        page_elems = []
        for r, t, size, bold in text_blocks:
            if any(_overlap_frac(r, tr) > 0.5 for tr, *_ in tables):
                continue
            if any(_overlap_frac(r, fr) > 0.8 for fr in fig_rects) and not CAPTION_RE.match(t):
                continue
            if self._is_heading(t, size, bold, body_size):
                section = t[:120]
            page_elems.append(Element("text", pno, t, tuple(r), section, origin="native"))
        out.elements.extend(page_elems)
        return section

    @staticmethod
    def _is_heading(t: str, size: float, bold: bool, body: float) -> bool:
        if len(t) > 120 or t.endswith((".", ",", ";", ":")) and not HEADING_NUM_RE.match(t):
            return False
        if size >= body * 1.15:
            return True
        return bold and len(t) < 90 and (HEADING_NUM_RE.match(t) is not None or t.istitle() or t.isupper())

    @staticmethod
    def _find_caption(r: pymupdf.Rect, text_blocks) -> str:
        best, best_d = "", 1e9
        for br, t, _, _ in text_blocks:
            if not CAPTION_RE.match(t):
                continue
            if min(br.x1, r.x1) - max(br.x0, r.x0) <= 0:  # no horizontal overlap
                continue
            d = br.y0 - r.y1 if br.y0 >= r.y1 - 5 else r.y0 - br.y1
            if -5 <= d < 80 and d < best_d:
                best, best_d = t, d
        return best

    def _vector_figures(self, page, tables, raster_rects) -> list[pymupdf.Rect]:
        try:
            drawings = page.get_drawings()
        except Exception:
            return []
        if len(drawings) < 3:
            return []
        try:
            clusters = page.cluster_drawings(drawings=drawings, x_tolerance=6, y_tolerance=6)
        except Exception:
            return []
        prect = page.rect
        out = []
        for c in clusters:
            r = pymupdf.Rect(c) & prect
            if r.is_empty or r.width < 0.18 * prect.width or r.height < 50:
                continue
            if r.get_area() > 0.85 * prect.get_area():
                continue
            if any(_overlap_frac(r, tr) > 0.5 or _overlap_frac(tr, r) > 0.5 for tr, *_ in tables):
                continue
            if any(_overlap_frac(r, rr) > 0.5 for rr in raster_rects):
                continue
            n_draw = sum(1 for d in drawings if _overlap_frac(pymupdf.Rect(d["rect"]), r) > 0.9)
            inner_text = page.get_text("text", clip=r)
            if n_draw < 4 and len(inner_text) > 300:  # a framed text box, not a figure
                continue
            out.append((r + (-8, -8, 8, 8)) & prect)
        return out
