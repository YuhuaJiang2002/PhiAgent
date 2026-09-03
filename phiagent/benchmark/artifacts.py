"""Content-addressed artifacts for resumable PhiAgent-Bench runs."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    sha256: str
    bytes: int
    cas_path: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class ArtifactStore:
    """Append-only local CAS; callers retain their original job outputs."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, path: Path) -> ArtifactRecord:
        source = path.expanduser().resolve()
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"artifact is missing or empty: {source}")
        digest = sha256_file(source)
        destination = self.root / digest[:2] / digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ValueError(f"CAS digest collision or corruption: {destination}")
        else:
            temporary = destination.with_suffix(
                f".{os.getpid()}.{threading.get_ident()}.tmp"
            )
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != digest:
                raise ValueError(f"artifact changed while being copied: {source}")
            temporary.replace(destination)
        return ArtifactRecord(
            path=str(source),
            sha256=digest,
            bytes=source.stat().st_size,
            cas_path=str(destination),
        )

    @staticmethod
    def verify(record: ArtifactRecord) -> bool:
        path = Path(record.path)
        cas = Path(record.cas_path)
        return (
            path.is_file()
            and cas.is_file()
            and path.stat().st_size == record.bytes
            and cas.stat().st_size == record.bytes
            and sha256_file(path) == record.sha256
            and sha256_file(cas) == record.sha256
        )
