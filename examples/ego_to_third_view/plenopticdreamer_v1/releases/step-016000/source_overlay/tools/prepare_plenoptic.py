#!/usr/bin/env python3
import plenoptic_paths as layout
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import venv

HERE = Path(__file__).resolve().parent
# Always follow the selected source tree, even if another project's activation
# variables remain in the caller's shell.
ROOT = layout.ROOT
CACHE_ROOT = layout.CACHE_ROOT
REPO = layout.rooted("cosmos-transfer2.5")
ENV = layout.rooted("tools/download-env")
PYTHON = ENV / "bin/python"

_cache_paths = {
    "HF_HOME": "huggingface",
    "HF_HUB_CACHE": "huggingface/hub",
    "HF_XET_CACHE": "huggingface/xet",
    "HF_MODULES_CACHE": "huggingface/modules",
    "PIP_CACHE_DIR": "pip",
    "UV_CACHE_DIR": "uv",
    "UV_PYTHON_INSTALL_DIR": "uv-python",
    "UV_PYTHON_BIN_DIR": "uv-python-bin",
    "UV_TOOL_DIR": "uv-tools",
    "UV_TOOL_BIN_DIR": "uv-bin",
    "XDG_CACHE_HOME": "xdg",
    "XDG_CONFIG_HOME": "config",
    "XDG_DATA_HOME": "data",
    "XDG_STATE_HOME": "state",
    "TORCH_HOME": "torch",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "PYTORCH_KERNEL_CACHE_PATH": "torch-kernels",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda",
    "NUMBA_CACHE_DIR": "numba",
    "MPLCONFIGDIR": "matplotlib",
    "GRADIO_TEMP_DIR": "gradio",
    "NLTK_DATA": "nltk",
    "WANDB_CACHE_DIR": "wandb/cache",
    "WANDB_CONFIG_DIR": "wandb/config",
    "WANDB_DATA_DIR": "wandb/data",
    "PYTHONPYCACHEPREFIX": "pycache",
    "TMPDIR": "temp",
    "TMP": "temp",
    "TEMP": "temp",
}
CACHE_ENV = {key: str(CACHE_ROOT / relative) for key, relative in _cache_paths.items()}
CACHE_ENV.update({
    "PLENOPTIC_ROOT": str(ROOT),
    "PLENOPTIC_CACHE_ROOT": str(CACHE_ROOT),
    "PLENOPTIC_SHARED_CACHE_ROOT": str(layout.SHARED_CACHE_ROOT),
    "PLENOPTIC_TRAIN_ENV": str(layout.TRAIN_ENV),
    "PLENOPTIC_DOWNLOAD_ENV": str(layout.DOWNLOAD_ENV),
    "PLENOPTIC_DATASETS_ROOT": str(layout.DATASETS_ROOT),
    "PLENOPTIC_OUTPUTS_ROOT": str(layout.OUTPUTS_ROOT),
    "PLENOPTIC_CHECKPOINTS_ROOT": str(layout.CHECKPOINTS_ROOT),
    "PLENOPTIC_STATE_ROOT": str(layout.STATE_ROOT),
    "UV_PROJECT_ENVIRONMENT": str(layout.TRAIN_ENV),
    "HF_TOKEN_PATH": str(CACHE_ROOT / "huggingface/token"),
})
os.environ.update(CACHE_ENV)
sys.pycache_prefix = str(CACHE_ROOT / "pycache")

_defaults = {
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "HF_HUB_DOWNLOAD_TIMEOUT": "120",
    "HF_HUB_ETAG_TIMEOUT": "60",
}
for _key, _value in _defaults.items():
    os.environ.setdefault(_key, _value)

# 固定版本，重跑时不会切换到更新的权重或数据。
JOBS = {
    "base": (
        "nvidia/Cosmos-Predict2.5-2B",
        "model",
        "15a82a2ec231bc318692aa0456a36537c806e7d4",
        "../checkpoints/Cosmos-Predict2.5-2B",
        [
            "base/post-trained/"
            "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
        ],
    ),
    "vae": (
        "nvidia/Cosmos-Predict2.5-2B",
        "model",
        "f176dc95b4a70f53ce01c4b302851595e7322b00",
        "../checkpoints/Cosmos-Predict2.5-2B",
        ["tokenizer.pth"],
    ),
    "reason": (
        "nvidia/Cosmos-Reason1-7B",
        "model",
        "3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
        "../checkpoints/Cosmos-Reason1-7B",
        [
            *[
                f"model-{i:05d}-of-00004.safetensors"
                for i in range(1, 5)
            ],
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    ),
    "syncam": (
        "KlingTeam/SynCamVideo-Dataset",
        "dataset",
        "74e4fcaf4f20981ed67c2c129a50f01fb600e5c4",
        "../DATASETS/symcam/raw",
        ["README.md", "SynCamVideo-Dataset.tar.gz"],
    ),
    "multicam": (
        "KlingTeam/MultiCamVideo-Dataset",
        "dataset",
        "4e7d995480cfa915f7c3721e616ed14789d87ff7",
        "../DATASETS/multicam/raw",
        [
            "README.md",
            *[
                f"MultiCamVideo-Dataset.parta{suffix}"
                for suffix in "abcdefghijklmnop"
            ],
        ],
    ),
}


def run(command):
    subprocess.run(command, check=True)


def output(command):
    return subprocess.check_output(command, text=True).strip()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )
    temporary.replace(path)


def retry(function, attempts=20):
    for attempt in range(1, attempts + 1):
        try:
            return function()
        except Exception as exc:
            status = getattr(
                getattr(exc, "response", None), "status_code", None
            )
            # 权限、文件不存在等问题，重试通常没有意义。
            if (
                status is not None
                and 400 <= status < 500
                and status not in (408, 429)
            ):
                raise
            # 磁盘空间不足、文件权限等本地错误直接停止。
            if isinstance(exc, OSError) and exc.errno is not None:
                raise
            if attempt == attempts:
                raise
            print(
                f"{type(exc).__name__}, HTTP={status}；"
                f"30 秒后重试（{attempt}/{attempts}）",
                flush=True,
            )
            time.sleep(30)


def setup():
    for relative in set(_cache_paths.values()):
        (CACHE_ROOT / relative).mkdir(parents=True, exist_ok=True)
    ENV.parent.mkdir(parents=True, exist_ok=True)
    ready = False
    if PYTHON.is_file() and os.access(PYTHON, os.X_OK):
        try:
            probe = subprocess.run(
                [str(PYTHON), "-c", "import sys, pip; print(sys.prefix)"],
                text=True, capture_output=True, timeout=20,
            )
            ready = probe.returncode == 0 and Path(probe.stdout.strip()).resolve() == ENV.resolve()
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not ready:
        if ENV.exists() or ENV.is_symlink():
            backup = ENV.with_name(ENV.name + ".backup-" + datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
            ENV.rename(backup)
            print(f"已保留无法使用的旧虚拟环境：{backup}", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENV)

    command = [
        str(PYTHON), "-m", "pip", "install",
        "--retries", "20", "--timeout", "60",
    ]
    help_text = output([
        str(PYTHON), "-m", "pip", "install", "--help"
    ])
    if "--resume-retries" in help_text:
        command += ["--resume-retries", "20"]

    command += ["huggingface_hub[cli]==0.36.0"]
    run(command)
    print(f"下载环境准备完成：{ENV}")


def use_download_environment():
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise RuntimeError("下载环境缺失或不可执行；请执行：python3 prepare_plenoptic.py setup。迁移项目后需要重新创建虚拟环境。")

    if Path(sys.prefix).resolve() != ENV.resolve():
        os.execv(
            str(PYTHON),
            [str(PYTHON), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        )


def download_directory(directory):
    """Keep Hugging Face's local-dir metadata and partial files in CACHE_ROOT."""
    destination = layout.rooted(directory)
    destination.mkdir(parents=True, exist_ok=True)
    local_cache = destination / ".cache/huggingface"
    cache_key = hashlib.sha256(layout.logical(destination).encode()).hexdigest()[:24]
    cache = layout.SHARED_CACHE_ROOT / "hf-local" / cache_key
    cache.parent.mkdir(parents=True, exist_ok=True)
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    if local_cache.is_symlink():
        if local_cache.resolve() != cache.resolve():
            raise RuntimeError(f"下载缓存链接指向其他目录：{local_cache}")
    else:
        if local_cache.exists():
            if cache.exists():
                raise RuntimeError(f"两个下载缓存目录同时存在，需先合并断点文件：{local_cache} 和 {cache}")
            local_cache.rename(cache)
        cache.mkdir(parents=True, exist_ok=True)
        local_cache.symlink_to(os.path.relpath(cache, local_cache.parent), target_is_directory=True)
    cache.mkdir(parents=True, exist_ok=True)
    return destination


def download(name):
    from huggingface_hub import get_token, hf_hub_download

    repo, kind, revision, directory, filenames = JOBS[name]
    destination = download_directory(directory)

    # 公共数据下载不携带登录 Token。
    token = (get_token() or False) if kind == "model" else False

    print(f"\n{name}: {repo}@{revision}", flush=True)

    for filename in filenames:
        print(f"下载/续传：{filename}", flush=True)

        downloaded = retry(lambda: hf_hub_download(
            repo_id=repo,
            repo_type=kind,
            revision=revision,
            filename=filename,
            local_dir=str(destination),
            token=token,
            force_download=False,
        ))

        path = Path(downloaded)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"文件未完整取得：{filename}")

    save_json(layout.rooted("download-state") / f"{name}.json", {
        "repo": repo,
        "revision": revision,
        "directory": str(layout.relative(destination)),
        "files": filenames,
        "status": "downloaded",
    })
    print(f"下载完成：{destination}", flush=True)


def image(build, minimum_free_gib):
    docker = ["sudo", "-n", "docker"]
    dockerfile = REPO / os.environ.get("PLENOPTIC_DOCKERFILE", "Dockerfile")

    if not dockerfile.is_file():
        raise RuntimeError(f"没有找到：{dockerfile}")

    print(f"源码目录：{REPO}")
    print(f"Git 元数据存在：{(REPO / '.git').exists()}")

    # 没有 .git 也能检查源码、构建镜像。
    camera_root = REPO / "cosmos_transfer2/_src/predict2/camera"
    for relative in [
        "inference/multiview_camera_ar_video2world.py",
        "datasets/camera_conditioned/dataset_utils.py",
    ]:
        path = camera_root / relative
        print(f"{'存在' if path.is_file() else '缺少'}：{path}")

    storage = output([
        *docker, "info", "--format", "{{.DockerRootDir}}"
    ])
    print(f"\nDocker 存储目录：{storage}")
    run(["sudo", "-n", "df", "-h", storage])
    run(["df", "-h", str(ROOT)])

    if not build:
        print("\n以上仅预检。添加 --build 才会拉取并构建镜像。")
        return

    disk = output(["sudo", "-n", "df", "-Pk", storage])
    free_bytes = int(disk.splitlines()[-1].split()[3]) * 1024
    if free_bytes < minimum_free_gib * 1024**3:
        raise RuntimeError(
            f"Docker 盘剩余 {free_bytes / 1024**3:.1f} GiB，"
            f"低于脚本预留值 {minimum_free_gib:g} GiB"
        )

    match = re.search(
        r"^ARG\s+BASE_IMAGE=(\S+)\s*$",
        dockerfile.read_text(),
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError("无法识别 Dockerfile 中的 BASE_IMAGE")
    base = match.group(1)

    # 第一次成功拉取后固定 digest，以后重跑复用同一版本。
    pin_file = layout.rooted("download-state/docker-base.json")
    if pin_file.exists():
        pin = json.loads(pin_file.read_text())
        if pin["source"] != base:
            raise RuntimeError("Dockerfile 的基础镜像与已有版本记录不同")
        pull_reference = pin["reference"]
    else:
        pull_reference = base

    retry(lambda: run([*docker, "pull", pull_reference]), attempts=5)

    if not pin_file.exists():
        digests = json.loads(output([
            *docker, "image", "inspect", pull_reference,
            "--format", "{{json .RepoDigests}}",
        ]))
        repository = base.split("@")[0]
        if ":" in repository.rsplit("/", 1)[-1]:
            repository = repository.rsplit(":", 1)[0]

        matches = [
            item for item in (digests or [])
            if item.startswith(repository + "@sha256:")
        ]
        if not matches:
            raise RuntimeError("无法取得基础镜像的固定 digest")

        pin = {"source": base, "reference": matches[0]}
        save_json(pin_file, pin)

    tag = "llq/plenoptic:local"
    run([
        "sudo", "-n", "env", "DOCKER_BUILDKIT=1",
        "docker", "build",
        "--progress=plain",
        "--build-arg", "BASE_IMAGE=" + pin["reference"],
        "-f", str(dockerfile),
        "-t", tag,
        str(REPO),
    ])

    image_id = output([
        *docker, "image", "inspect", tag, "--format", "{{.Id}}"
    ])
    save_json(layout.rooted("image-build.json"), {
        "image": tag,
        "image_id": image_id,
        "base": pin["reference"],
        "source_directory": str(REPO),
    })
    print(f"镜像构建完成：{tag}")



def save_download_token_for_mirror():
    from getpass import getpass
    from huggingface_hub.constants import HF_TOKEN_PATH

    print("模型下载将通过第三方 hf-mirror.com，并携带此 Token。")
    token = getpass("HF Read Token（输入不显示）: ").strip()
    if not token:
        raise RuntimeError("未输入 Token")

    token_path = Path(HF_TOKEN_PATH)
    token_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor = os.open(
        token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
    )
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(token)

    print("Token 已保存到本地，尚未联网验证。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tasks", nargs="+",
        choices=["setup", "env", "login", "image", "weights", *JOBS],
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=100)
    args = parser.parse_args()

    if args.min_free_gib <= 0:
        parser.error("--min-free-gib 必须大于 0")

    if args.tasks == ["env"]:
        layout.STATE_ROOT.mkdir(parents=True, exist_ok=True)
        layout.SHARED_STATE_ROOT.mkdir(parents=True, exist_ok=True)
        for relative in set(_cache_paths.values()):
            (CACHE_ROOT / relative).mkdir(parents=True, exist_ok=True)
        for key, value in {**CACHE_ENV, **{key: os.environ[key] for key in _defaults}}.items():
            print(f"export {key}={shlex.quote(value)}")
        return
    if "env" in args.tasks:
        parser.error("env 请单独执行")

    # setup 单独运行，避免重入时重复安装。
    if "setup" in args.tasks:
        if args.tasks != ["setup"]:
            parser.error("setup 请单独执行")
        setup()
        return

    if args.tasks == ["image"]:
        image(args.build, args.min_free_gib)
        return
    if "image" in args.tasks:
        parser.error("image 请单独执行")

    use_download_environment()

    if "login" in args.tasks:
        from huggingface_hub import login
        save_download_token_for_mirror()

    selected = []
    for name in args.tasks:
        if name == "weights":
            selected.extend(["base", "vae", "reason"])
        elif name != "login":
            selected.append(name)

    for name in dict.fromkeys(selected):
        download(name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。保留缓存，重跑原命令继续下载。")
        sys.exit(130)
