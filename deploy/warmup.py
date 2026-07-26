from __future__ import annotations

import json

import numpy as np

from deploy.preflight import run_preflight
from ocr.paddle_ocr import PaddleOcr


def main() -> None:
    runtime = run_preflight()
    image = np.full((64, 256, 3), 255, dtype=np.uint8)
    device = "gpu:0" if runtime.runtime == "gpu" else "cpu"
    ocr = PaddleOcr(device=device)
    pages = ocr.predict(image)
    if len(pages) != 1:
        raise RuntimeError(
            f"OCR warmup returned {len(pages)} pages; expected exactly one"
        )
    print(
        json.dumps(
            {
                "status": "warmed",
                "runtime": runtime.runtime,
                "visible_gpu_count": runtime.visible_gpu_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
