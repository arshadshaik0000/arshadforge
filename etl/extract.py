"""
ETL Extract Module — PDF text + table extraction with pdfplumber.

Parses the Cyber Ireland 2022 Report into structured elements:
- Text blocks with page numbers, section headings
- Tables as structured {columns, rows, page} JSON

No LLM calls. Pure deterministic parsing.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

import pdfplumber

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\x20-\x7E\u00C0-\u024F€£¥°±²³µ·¹º»¼½¾\n]", "", text)
    return text


def _is_ocr_garbage(text: str) -> bool:
    """Detect OCR noise — lines with >50% single-char tokens."""
    tokens = text.split()
    if len(tokens) < 5:
        return False
    single_char = sum(1 for t in tokens if len(t) == 1)
    return single_char / len(tokens) > 0.5


def _detect_heading(line: str) -> Optional[str]:
    """Heuristic heading detection: numbered sections or ALL-CAPS lines."""
    stripped = line.strip()
    if not stripped or len(stripped) < 4:
        return None

    # Numbered section headings: "3.2 NUMBER OF FIRMS"
    if re.match(r"^\d+\.\d+\s+[A-Z]", stripped):
        return stripped

    # All-caps lines longer than 10 chars (likely section titles)
    if stripped.isupper() and len(stripped) > 10 and " " in stripped:
        return stripped

    return None


def _clean_header(header: Optional[str]) -> str:
    """Normalize a single table header string."""
    if not header:
        return ""
    header = re.sub(r"\s+", " ", str(header)).strip()
    if len(header) > 120:
        return ""
    return header


def _clean_headers(headers: list[Optional[str]]) -> list[str]:
    """Clean all headers; if any is too long, use Column_N fallback."""
    cleaned = [_clean_header(h) for h in headers]
    if all(not h for h in cleaned):
        return [f"Column_{i+1}" for i in range(len(headers))]
    # Replace empty headers with positional names
    for i, h in enumerate(cleaned):
        if not h:
            cleaned[i] = f"Column_{i+1}"
    return cleaned


def _parse_table(raw_table: list[list], page_num: int) -> Optional[dict[str, Any]]:
    """Convert a pdfplumber raw table into a structured dict."""
    if not raw_table or len(raw_table) < 2:
        return None

    # First row is headers
    headers = _clean_headers(raw_table[0])

    rows = []
    for raw_row in raw_table[1:]:
        # Skip rows that are all None or empty
        if all(not cell for cell in raw_row):
            continue
        row = {}
        for i, cell in enumerate(raw_row):
            if i < len(headers):
                row[headers[i]] = _clean_text(str(cell)) if cell else ""
        if any(v for v in row.values()):
            rows.append(row)

    if not rows:
        return None

    return {
        "type": "table",
        "page": page_num,
        "columns": headers,
        "rows": rows,
    }


def extract_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text and table elements from a PDF file.

    Returns:
        List of element dicts, each with:
        - type: "text" or "table"
        - page: int (1-indexed)
        - content: str (for text) or columns/rows (for table)
        - section: str (detected heading)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    elements: list[dict[str, Any]] = []
    current_section = "Introduction"

    logger.info(f"Extracting PDF: {file_path.name}")

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"Total pages: {total_pages}")

        for page in pdf.pages:
            page_num = page.page_number
            text = page.extract_text() or ""

            if _is_ocr_garbage(text):
                logger.warning(f"Page {page_num}: Detected OCR garbage, skipping text")
                text = ""

            # Detect headings in text
            if text:
                for line in text.split("\n"):
                    heading = _detect_heading(line)
                    if heading:
                        current_section = heading

                # Emit text element
                cleaned = _clean_text(text)
                if len(cleaned) > 50:  # Skip very short pages (headers/footers only)
                    elements.append({
                        "type": "text",
                        "page": page_num,
                        "content": cleaned,
                        "section": current_section,
                    })

            # Extract tables
            raw_tables = page.extract_tables() or []
            for raw_table in raw_tables:
                table_elem = _parse_table(raw_table, page_num)
                if table_elem:
                    table_elem["section"] = current_section
                    elements.append(table_elem)

    text_count = sum(1 for e in elements if e["type"] == "text")
    table_count = sum(1 for e in elements if e["type"] == "table")
    logger.info(f"Extracted {text_count} text blocks, {table_count} tables from {total_pages} pages")

    return elements
