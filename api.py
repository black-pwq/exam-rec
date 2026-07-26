from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app_logging import configure_logging
from recognition_jobs import (
    DefaultRecognitionProcessor,
    JobNotCompleteError,
    JobNotFoundError,
    JobQueueFullError,
    JobStore,
    LlmSettings,
    RecognitionService,
)


@dataclass(frozen=True)
class ApiSettings:
    job_root: Path = Path("var/jobs")
    max_upload_bytes: int = 500 * 1024 * 1024
    max_pdf_pages: int = 500
    max_queued_jobs: int = 32

    def __post_init__(self) -> None:
        if self.max_upload_bytes < 1:
            raise ValueError("max_upload_bytes must be positive")
        if self.max_pdf_pages < 1:
            raise ValueError("max_pdf_pages must be positive")
        if self.max_queued_jobs < 1:
            raise ValueError("max_queued_jobs must be positive")

    @classmethod
    def from_env(cls) -> ApiSettings:
        return cls(
            job_root=Path(os.getenv("EXAM_REC_JOB_ROOT", "var/jobs")),
            max_upload_bytes=int(
                os.getenv("EXAM_REC_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024))
            ),
            max_pdf_pages=int(os.getenv("EXAM_REC_MAX_PDF_PAGES", "500")),
            max_queued_jobs=int(os.getenv("EXAM_REC_MAX_QUEUED_JOBS", "32")),
        )


def build_default_service(settings: ApiSettings) -> RecognitionService:
    store = JobStore(settings.job_root)
    return RecognitionService(
        store,
        lambda: DefaultRecognitionProcessor(LlmSettings.from_env()),
        max_queued_jobs=settings.max_queued_jobs,
    )


def create_app(
    *,
    service: RecognitionService | None = None,
    settings: ApiSettings | None = None,
) -> FastAPI:
    configure_logging()
    configured = settings or ApiSettings.from_env()
    supplied_service = service

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        recognition = supplied_service or build_default_service(configured)
        await run_in_threadpool(recognition.start)
        app.state.recognition_service = recognition
        try:
            yield
        finally:
            await run_in_threadpool(recognition.stop)

    app = FastAPI(
        title="Exam Recognition API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.api_settings = configured

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        recognition = _service(request)
        return {"status": "ok" if recognition.running else "unavailable"}

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready(request: Request) -> Response:
        recognition = _service(request)
        if recognition.running:
            return JSONResponse({"status": "ready"})
        return JSONResponse(
            {"status": "unavailable"},
            status_code=503,
        )

    @app.post("/recognitions", status_code=202)
    async def create_recognition(
        request: Request,
        response: Response,
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        recognition = _service(request)
        job_id = recognition.store.new_job_id()
        upload_path = recognition.store.prepare_upload(job_id)
        try:
            size = await run_in_threadpool(
                _copy_upload,
                file,
                upload_path,
                configured.max_upload_bytes,
            )
            page_count = await run_in_threadpool(
                _validate_pdf,
                upload_path,
                configured.max_pdf_pages,
            )
            status = recognition.store.create(
                job_id,
                original_filename=Path(file.filename or "input.pdf").name,
                page_count=page_count,
            )
            try:
                recognition.submit(job_id)
            except JobQueueFullError as error:
                recognition.store.delete(job_id)
                raise HTTPException(
                    status_code=429,
                    detail=str(error),
                    headers={"Retry-After": "30"},
                ) from error
            except RuntimeError as error:
                recognition.store.delete(job_id)
                raise HTTPException(
                    status_code=503,
                    detail="recognition worker is unavailable",
                ) from error
        except HTTPException:
            recognition.store.discard_upload(job_id)
            raise
        except Exception as error:
            recognition.store.discard_upload(job_id)
            raise HTTPException(
                status_code=422,
                detail=f"invalid PDF upload: {error}",
            ) from error
        finally:
            await file.close()

        response.headers["Location"] = f"/recognitions/{job_id}"
        return {
            "job_id": job_id,
            "status": status["status"],
            "file_size": size,
            "page_count": page_count,
        }

    @app.get("/recognitions/{job_id}")
    async def get_recognition(job_id: str, request: Request) -> dict[str, Any]:
        try:
            return _service(request).store.get_status(job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.get("/recognitions/{job_id}/updates")
    async def get_recognition_updates(
        job_id: str,
        request: Request,
        after: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
    ) -> dict[str, Any]:
        try:
            return _service(request).store.get_updates(
                job_id,
                after=after,
                limit=limit,
            )
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    @app.get("/recognitions/{job_id}/result")
    async def get_recognition_result(
        job_id: str,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return _service(request).store.get_result(job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="job not found") from error
        except JobNotCompleteError as error:
            raise HTTPException(
                status_code=409,
                detail={"message": str(error), "status": error.status},
            ) from error

    @app.delete("/recognitions/{job_id}", status_code=202)
    async def delete_recognition(
        job_id: str,
        request: Request,
    ) -> dict[str, Any]:
        try:
            deleted = _service(request).delete(job_id)
        except JobNotFoundError as error:
            raise HTTPException(status_code=404, detail="job not found") from error
        return {
            "job_id": job_id,
            "status": "deleted" if deleted else "cancelling",
        }

    return app


def _service(request: Request) -> RecognitionService:
    return request.app.state.recognition_service


def _copy_upload(
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
) -> int:
    size = 0
    with destination.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDF exceeds the {max_bytes}-byte upload limit",
                )
            output.write(chunk)
    if size == 0:
        raise HTTPException(status_code=422, detail="uploaded PDF is empty")
    return size


def _validate_pdf(path: Path, max_pages: int) -> int:
    try:
        document = pymupdf.open(path)
    except Exception as error:
        raise HTTPException(status_code=422, detail="uploaded file is not a PDF") from error
    try:
        if document.needs_pass:
            raise HTTPException(
                status_code=422,
                detail="password-protected PDFs are not supported",
            )
        if not document.is_pdf or document.page_count == 0:
            raise HTTPException(
                status_code=422,
                detail="uploaded file must be a non-empty PDF",
            )
        if document.page_count > max_pages:
            raise HTTPException(
                status_code=413,
                detail=f"PDF exceeds the {max_pages}-page limit",
            )
        return document.page_count
    finally:
        document.close()


app = create_app()
