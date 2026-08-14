#!/usr/bin/env python3
"""Generate and independently decode-verify a repository QR code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse


def validate_github_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError("QR target must be an https://github.com repository URL")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise ValueError("QR target must identify exactly one GitHub owner/repository")
    return f"https://github.com/{parts[0]}/{parts[1]}"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--box-size", type=int, default=16)
    parser.add_argument("--border", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    target = validate_github_url(args.url)
    try:
        import cv2
        import qrcode
    except ImportError as exc:
        raise SystemExit("Install optional qrcode[pil] and OpenCV dependencies") from exc

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=args.box_size,
        border=args.border,
    )
    qr.add_data(target)
    qr.make(fit=True)
    qr.make_image(fill_color="black", back_color="white").save(output)

    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
    decoded, points, _ = cv2.QRCodeDetector().detectAndDecode(image)
    if points is None or decoded != target:
        raise RuntimeError(f"QR verification failed: decoded={decoded!r}, expected={target!r}")

    manifest = {
        "schema_version": "1.0.0",
        "target": target,
        "decoded_target": decoded,
        "decode_verified": True,
        "error_correction": "H",
        "border_modules": args.border,
        "sha256": sha256_file(output),
    }
    manifest_path = (args.manifest or output.with_suffix(".json")).expanduser().resolve()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"image": str(output), "manifest": str(manifest_path), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
