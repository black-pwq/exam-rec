from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_runtime_image_uses_namespaced_source_and_unprivileged_user() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "libgl1" in dockerfile
    assert 'import cv2; print(' in dockerfile
    assert 'PYTHONPATH="/app/src"' in dockerfile
    assert "COPY --link --chown=10001:10001 src ./src" in dockerfile
    assert "USER examrec" in dockerfile
    assert "exam_rec.main:app" in dockerfile
    assert "/health/ready" in dockerfile
    assert "deploy/" not in dockerfile
    assert "VCS_REF" not in dockerfile


def test_compose_uses_the_project_environment_and_preserves_runtime_options() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    gpu = (ROOT / "compose.gpu.yaml").read_text(encoding="utf-8")

    assert "- ./.env" in compose
    assert (
        '"${EXAM_REC_BIND_IP:-127.0.0.1}:${EXAM_REC_PORT:-8080}:8000"'
        in compose
    )
    assert "restart: unless-stopped" in compose
    assert "driver: json-file" in compose
    assert "paddle-cu118" in gpu
    assert "capabilities:" in gpu


def test_environment_example_is_safe_and_real_environment_is_ignored() -> None:
    environment = (ROOT / ".env.example").read_text(encoding="utf-8")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "EXAM_REC_BIND_IP=127.0.0.1" in environment
    assert "EXAM_REC_LLM_API_KEY=replace-with-the-LLM-api-key" in environment
    assert ".env\n" in gitignore
    assert "!.env.example" in gitignore
    assert ".env\n" in dockerignore


def test_legacy_top_level_application_modules_are_removed() -> None:
    for path in (
        "api.py",
        "app_logging.py",
        "pipeline.py",
        "question_range.py",
        "recognition_jobs.py",
        "transform.py",
        "extractor",
        "ocr",
        "utils",
        "deploy",
    ):
        assert not (ROOT / path).exists()
