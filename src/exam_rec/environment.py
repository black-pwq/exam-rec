from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_environment(path: str | Path | None = None) -> bool:
    dotenv_path = Path(path) if path is not None else Path.cwd() / ".env"
    return load_dotenv(dotenv_path=dotenv_path, override=False)
