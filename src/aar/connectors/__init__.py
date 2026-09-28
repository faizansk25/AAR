"""Connectors: how data gets into and out of AAR."""

from .excel import (  # noqa: F401
    detect_header_row, excel_to_canonical, read_excel, write_excel,
)

__all__ = [
    "detect_header_row", "excel_to_canonical", "read_excel", "write_excel",
]
