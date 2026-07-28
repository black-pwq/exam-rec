from collections.abc import Iterable
from os import PathLike

from exam_rec.ocr.page_ocr import PdfPageSource


def select_pdf_pages(
    path: str | PathLike[str],
    page_indexes: Iterable[int],
) -> bytes:
    """Return a PDF containing the requested pages in the requested order."""
    return PdfPageSource(path).select_pages(tuple(page_indexes))
