# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "docling",
# ]
# ///

import sys
from pathlib import Path

from docling.document_converter import DocumentConverter


def main():
    if len(sys.argv) != 2:
        print("Usage: process_pdf.py <path_to_pdf>", file=sys.stderr)
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    current_page = None
    parts = []

    for item, _level in doc.iterate_items():
        page_no = None
        prov = getattr(item, "prov", None)
        if prov:
            page_no = prov[0].page_no

        if page_no is not None and page_no != current_page:
            parts.append(f"<!-- page {page_no} -->")
            current_page = page_no

        typename = type(item).__name__
        text = getattr(item, "text", None)
        if not text:
            if typename == "PictureItem":
                parts.append("<!-- image -->")
            continue

        if typename == "SectionHeaderItem":
            level = getattr(item, "level", 1) or 1
            prefix = "#" * (level + 1)
            parts.append(f"{prefix} {text}")
        elif typename == "TableItem":
            table_md = getattr(item, "export_to_markdown", None)
            if callable(table_md):
                parts.append(table_md())
            else:
                parts.append(text)
        else:
            parts.append(text)

    sys.stdout.write("\n\n".join(parts))


if __name__ == "__main__":
    main()
