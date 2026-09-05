"""Dependency-free guard for the archived experiment adapters."""
import os
from pathlib import Path


def require_launcher():
    record = os.environ.get('PHI_EGO_OUTPUT')
    if not record or not (Path(record) / 'execution.json').is_file():
        raise SystemExit('Use examples/ego_to_third_view/pipeline.py; direct stage execution is disabled.')
