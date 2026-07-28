from __future__ import annotations

from exam_rec.api import create_app
from exam_rec.environment import load_project_environment


load_project_environment()
app = create_app()
