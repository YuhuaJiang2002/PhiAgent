#!/usr/bin/env python3
"""Small OpenAI-compatible text/image/video API probe.

API keys are read from environment variables and are never written to disk or
printed. Local media are converted to data URLs in memory and are not written
to disk. Run ``python3 test_llm_api.py --help`` for examples.
"""

from __future__ import annotations

import argparse
import json
import base64
import mimetypes
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


PROVIDERS = {
    "hunyuan": {
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "env_var": "HUNYUAN_API_KEY",
        "model_env": "HUNYUAN_MODEL",
        "model": "hunyuan-turbos-latest",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_var": "DASHSCOPE_API_KEY",
        "model_env": "DASHSCOPE_MODEL",
        "model": "qwen-plus",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_var": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "model": "gpt-5-mini",
    },
    "doubao": {
        # Ark's regular OpenAI-compatible endpoint.  Coding Plan users can
        # override this with ARK_BASE_URL in .env.
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "base_url_env": "ARK_BASE_URL",
        "env_var": "ARK_API_KEY",
        # Online inference commonly uses a deployed endpoint id (``ep-...``);
        # Coding Plan also accepts its documented model aliases.
        "model": None,
        "model_env": "ARK_MODEL",
    },
    "qwen38-local": {
        "base_url": "http://127.0.0.1:18080/v1",
        "env_var": None,
        "model": "qwen3.8-27b-fp8",
        "model_env": "QWEN38_MODEL",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
}



def load_dotenv(path: Path | None) -> None:
    """Load simple KEY=VALUE entries without overriding the environment.

    This intentionally supports only the small dotenv subset needed for API
    keys: blank lines, comments, optional ``export``, and single/double quoted
    values. It avoids adding a runtime dependency just to read credentials.
    """

    if path is None or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or not key.replace("_", "A").isalnum() or key[0].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    examples = """examples:
  export DASHSCOPE_API_KEY='your-key'
  python3 test_llm_api.py --provider dashscope

  export ARK_API_KEY='your-ark-key'
  export ARK_MODEL='ep-your-multimodal-endpoint'
  python3 test_llm_api.py --provider doubao --prompt '请描述图片' \
      --image /path/to/image.jpg
  python3 test_llm_api.py --provider doubao --prompt '视频中是否出现手' \
      --video /path/to/video.mp4

  python3 test_llm_api.py --provider qwen38-local --prompt '请描述图片' \
      --image /path/to/image.jpg
"""
    parser = argparse.ArgumentParser(
        description="Send one OpenAI-compatible text/image/video chat request.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parent / ".env",
        help="Optional dotenv file (default: project .env); shell variables win.",
    )
    parser.add_argument(
        "--model",
        help="Override the provider's default model name.",
    )
    parser.add_argument(
        "--base-url",
        help="Override the provider base URL (or its *_BASE_URL value in .env).",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: API connection successful",
        help="Prompt sent to the model.",
    )
    media = parser.add_mutually_exclusive_group()
    media.add_argument(
        "--image",
        help="Local image path or an http(s)/data URL.",
    )
    media.add_argument(
        "--video",
        help="Local video path or an http(s)/data URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the JSON payload without making an API call.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds (default: 60).",
    )
    return parser.parse_args()


def media_url(value: str) -> str:
    """Return a URL suitable for an OpenAI-compatible media content item."""
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme in {"http", "https", "data"}:
        return value

    path = Path(value)
    if not path.is_file():
        raise ValueError(f"media file does not exist: {value}")
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_content(prompt: str, image: str | None, video: str | None) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image:
        content.append({"type": "image_url", "image_url": {"url": media_url(image)}})
    if video:
        content.append({"type": "video_url", "video_url": {"url": media_url(video)}})
    return content


def payload_summary(payload: dict) -> dict:
    """Return a safe summary without printing base64 media or secrets."""
    content = payload["messages"][0]["content"]
    items = []
    for item in content:
        if item["type"] == "text":
            items.append({"type": "text", "chars": len(item["text"])})
        else:
            key = "image_url" if item["type"] == "image_url" else "video_url"
            url = item[key]["url"]
            items.append({
                "type": item["type"],
                "url_kind": "data_url" if url.startswith("data:") else "remote_url",
                "url_chars": len(url),
            })
    return {"model": payload["model"], "content": items, "stream": payload["stream"]}


def extract_text(response: dict) -> str | None:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "".join(parts) or None
    return None


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file.expanduser().resolve() if args.env_file else None)
    config = PROVIDERS[args.provider]
    env_var = config.get("env_var")
    api_key = os.environ.get(env_var) if env_var else None
    model_env = config.get("model_env")
    model = args.model or (os.environ.get(model_env) if model_env else None) or config.get("model")
    if not model:
        hint = f" or {model_env}" if model_env else ""
        print(f"Error: --model{hint} is required for provider {args.provider}.", file=sys.stderr)
        return 2

    try:
        content = build_content(args.prompt, args.image, args.video)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }
    payload.update(config.get("extra_body", {}))
    if args.dry_run:
        print(json.dumps(payload_summary(payload), ensure_ascii=False, indent=2))
        return 0

    if env_var and not api_key:
        print(f"Error: environment variable {env_var} is not set.", file=sys.stderr)
        print(f"Set it temporarily with: export {env_var}='your-key'", file=sys.stderr)
        return 2

    base_url_env = config.get("base_url_env")
    base_url = args.base_url or (os.environ.get(base_url_env) if base_url_env else None) or config["base_url"]
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "llm-api-connectivity-test/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    print(f"provider={args.provider} model={model} media={('image' if args.image else 'video' if args.video else 'text')}")
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {error_body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Connection error: {exc.reason}", file=sys.stderr)
        return 1
    except TimeoutError:
        print(f"Connection timed out after {args.timeout:g} seconds.", file=sys.stderr)
        return 1

    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        print(f"HTTP {status}, but the response was not JSON:\n{body}", file=sys.stderr)
        return 1

    print(f"HTTP {status}")
    text = extract_text(result)
    if text is not None:
        print(text)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
