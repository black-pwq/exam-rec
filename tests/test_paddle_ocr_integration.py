from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from ocr.paddle_ocr import PaddleOcr


@pytest.mark.integration
def test_real_paddle_ocr_recognizes_generated_image(tmp_path: Path) -> None:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font_path.exists():
        pytest.skip("DejaVu Sans is unavailable")

    image_path = tmp_path / "ocr-input.png"
    image = Image.new("RGB", (720, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (30, 35),
        "PADDLE OCR 12345",
        fill="black",
        font=ImageFont.truetype(str(font_path), 72),
    )
    image.save(image_path)

    results = PaddleOcr().predict(image_path)

    assert results
    elements = [element for page in results for element in page]
    recognized = " ".join(element.content for element in elements).upper()
    assert "12345" in recognized
    assert all(element.bbox for element in elements)
