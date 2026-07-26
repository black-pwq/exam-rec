from pathlib import Path

import pymupdf
import pytest

from ocr.page_ocr import PdfPageSource
from utils.pdf import select_pdf_pages


def make_pdf(path: Path, texts: list[str]) -> None:
    document = pymupdf.open()
    for text in texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_select_pdf_pages_preserves_requested_order(tmp_path) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path, ["first", "second", "third"])

    selected = pymupdf.open(
        stream=select_pdf_pages(path, [2, 0]),
        filetype="pdf",
    )
    try:
        assert selected.page_count == 2
        assert [page.get_text().strip() for page in selected] == ["third", "first"]
    finally:
        selected.close()


def test_pdf_page_source_accepts_an_in_memory_snapshot(tmp_path) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path, ["first", "second"])

    source = PdfPageSource(path.read_bytes())
    selected = pymupdf.open(
        stream=source.select_pages([1, 0]),
        filetype="pdf",
    )
    try:
        assert source.page_count == 2
        assert [page.get_text().strip() for page in selected] == ["second", "first"]
    finally:
        selected.close()


def test_select_pdf_pages_validates_input(tmp_path) -> None:
    path = tmp_path / "input.pdf"
    make_pdf(path, ["only"])

    with pytest.raises(ValueError, match="must not be empty"):
        select_pdf_pages(path, [])
    with pytest.raises(ValueError, match="invalid PDF page index"):
        select_pdf_pages(path, [1])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        select_pdf_pages(tmp_path / "missing.pdf", [0])


def test_select_pdf_pages_rejects_password_protected_pdf(tmp_path) -> None:
    source = pymupdf.open()
    source.new_page()
    path = tmp_path / "protected.pdf"
    source.save(
        path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    source.close()

    with pytest.raises(ValueError, match="password protected"):
        select_pdf_pages(path, [0])
