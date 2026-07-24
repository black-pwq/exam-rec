from collections.abc import Iterable
from os import PathLike, fspath
from pathlib import Path

import pymupdf


def select_pdf_pages(
    path: str | PathLike[str],
    page_indexes: Iterable[int],
) -> bytes:
    """Return a PDF containing the requested pages in the requested order."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {source_path}")

    indexes = list(page_indexes)
    if not indexes:
        raise ValueError("page_indexes must not be empty")

    document = pymupdf.open(fspath(source_path))
    selected = pymupdf.open()
    try:
        if not document.is_pdf or document.page_count == 0:
            raise ValueError(f"input is not a non-empty PDF: {source_path}")
        if document.needs_pass:
            raise ValueError(f"PDF is password protected: {source_path}")
        if any(
            not isinstance(index, int)
            or index < 0
            or index >= document.page_count
            for index in indexes
        ):
            raise ValueError("page_indexes contains an invalid PDF page index")

        for page_index in indexes:
            selected.insert_pdf(
                document,
                from_page=page_index,
                to_page=page_index,
            )
        return selected.tobytes()
    finally:
        selected.close()
        document.close()
