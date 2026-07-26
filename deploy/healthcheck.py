from __future__ import annotations

import json
from urllib.request import urlopen


def main() -> None:
    with urlopen("http://127.0.0.1:8000/health/ready", timeout=5) as response:
        body = json.load(response)
        if response.status != 200 or body != {"status": "ready"}:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
