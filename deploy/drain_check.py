from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


ACTIVE_STATUSES = {"queued", "running", "cancelling"}


def find_active_jobs(root: Path) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    if not root.exists():
        return active
    for status_path in sorted(root.glob("*/status.json")):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"failed to read job status: {status_path}: {error}"
            ) from error
        if status.get("status") in ACTIVE_STATUSES:
            active.append(
                {
                    "job_id": status.get("job_id", status_path.parent.name),
                    "status": status["status"],
                }
            )
    return active


def wait_until_drained(
    root: Path,
    *,
    timeout: float,
    interval: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while active := find_active_jobs(root):
        print(json.dumps({"status": "waiting", "active_jobs": active}))
        if time.monotonic() >= deadline:
            return active
        time.sleep(interval)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail while recognition jobs are still active.",
    )
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.timeout < 0 or args.interval <= 0:
        parser.error("timeout must be non-negative and interval must be positive")

    root = Path(os.getenv("EXAM_REC_JOB_ROOT", "var/jobs"))
    active = (
        wait_until_drained(
            root,
            timeout=args.timeout,
            interval=args.interval,
        )
        if args.wait
        else find_active_jobs(root)
    )
    if active:
        print(json.dumps({"status": "busy", "active_jobs": active}))
        raise SystemExit(1)
    print(json.dumps({"status": "drained", "active_jobs": []}))


if __name__ == "__main__":
    main()
