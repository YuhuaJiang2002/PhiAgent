#!/usr/bin/env python3
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time

ROOT = Path(os.environ.get('PLENOPTIC_ROOT', Path(__file__).resolve().parent.parent)).resolve()
STATE = ROOT / "datasets/prepare-state"


def fingerprint(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino]


def save(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )
    tmp.replace(path)


class Parts(io.RawIOBase):
    """按顺序读取分片，不生成合并后的大文件。"""

    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.handle = None
        self.total = sum(p.stat().st_size for p in paths)
        self.done = 0
        self.last = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.handle is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                print("读取分片：", path, flush=True)
                self.handle = path.open("rb")

            n = self.handle.readinto(buffer)
            if n:
                self.done += n
                if time.monotonic() - self.last > 30:
                    print(
                        f"输入扫描 {self.done / self.total:.1%} "
                        f"({self.done / 2**30:.1f}/"
                        f"{self.total / 2**30:.1f} GiB)",
                        flush=True,
                    )
                    self.last = time.monotonic()
                return n

            self.handle.close()
            self.handle = None

    def close(self):
        if self.handle is not None:
            self.handle.close()
        super().close()


def extract(name, paths, inputs):
    destination = ROOT / "datasets/extracted" / name
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()

    before_inputs = [fingerprint(p) for p in paths]
    identity = STATE / (name + ".inputs.json")
    if identity.exists() and json.loads(identity.read_text()) != inputs:
        raise RuntimeError(
            f"{name}: 原包发生变化，请先检查，不能混用旧解压断点"
        )
    save(identity, inputs)

    journal = STATE / (name + ".files.jsonl")
    completed = {}

    with journal.open("a+b") as log:
        log.seek(0)
        while True:
            offset = log.tell()
            line = log.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                log.truncate(offset)
                break
            entry = json.loads(line)
            completed[entry[0]] = entry[1:]
        log.seek(0, 2)

        written = skipped = 0
        with io.BufferedReader(Parts(paths), 1024 * 1024) as raw:
            stream = raw
            if raw.peek(4)[:4] == b"\x28\xb5\x2f\xfd":
                try:
                    import zstandard
                except ImportError:
                    raise RuntimeError(
                        "检测到 zstd，请给当前 Python 安装 "
                        "zstandard 后重启"
                    )
                stream = zstandard.ZstdDecompressor().stream_reader(
                    raw, read_across_frames=True
                )

            try:
                with tarfile.open(fileobj=stream, mode="r|*") as archive:
                    for member in archive:
                        relative = PurePosixPath(member.name)
                        if relative.is_absolute() or ".." in relative.parts:
                            raise RuntimeError(
                                f"非法归档路径：{member.name}"
                            )

                        target = destination.joinpath(*relative.parts)
                        if target.is_symlink():
                            raise RuntimeError(
                                f"输出位置已有软链接：{target}"
                            )
                        target.resolve().relative_to(destination)

                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        if not member.isfile():
                            raise RuntimeError(
                                f"不支持的归档成员：{member.name}"
                            )

                        key = relative.as_posix()
                        previous = completed.get(key, [])
                        can_skip = (
                            target.is_file()
                            and target.stat().st_size == member.size
                            and previous and previous[0] == fingerprint(target)
                        )
                        if not can_skip and target.is_file() and target.stat().st_size == member.size and len(previous) > 1:
                            # A copy to a new machine changes inode/ctime; compare content instead.
                            digest = hashlib.sha256()
                            with target.open('rb') as current:
                                for block in iter(lambda: current.read(4 * 1024 * 1024), b''):
                                    digest.update(block)
                            can_skip = digest.hexdigest() == previous[1]
                        if can_skip:
                            skipped += 1
                        else:
                            target.parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            temporary = target.with_name(
                                "." + target.name + ".extract-partial"
                            )
                            if temporary.is_symlink():
                                raise RuntimeError(
                                    f"临时路径已有软链接：{temporary}"
                                )

                            digest = hashlib.sha256()
                            with (
                                archive.extractfile(member) as src,
                                temporary.open("wb") as dst,
                            ):
                                for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
                                    digest.update(block)
                                    dst.write(block)

                            if temporary.stat().st_size != member.size:
                                raise RuntimeError(
                                    f"解压大小不符：{member.name}"
                                )

                            temporary.replace(target)
                            stamp = fingerprint(target)
                            log.write(
                                (json.dumps([key, stamp, digest.hexdigest()]) + "\n").encode()
                            )
                            log.flush()
                            completed[key] = [stamp, digest.hexdigest()]
                            written += 1

                        if (written + skipped) % 500 == 0:
                            print(
                                f"{name}: 新写 {written}，"
                                f"复用 {skipped} 个文件",
                                flush=True,
                            )
            finally:
                if stream is not raw:
                    stream.close()

    if before_inputs != [fingerprint(p) for p in paths]:
        raise RuntimeError(f"{name}: 解压期间原包发生变化")

    print(
        f"{name}: 解压结束，新写 {written}，复用 {skipped}",
        flush=True,
    )
    return destination


def index(name, destination):
    output = ROOT / "datasets/manifests"
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / (name + "_scenes.jsonl")
    tmp = manifest.with_suffix(".jsonl.tmp")
    counts = {
        "scenes": 0,
        "structurally_complete": 0,
        "incomplete": 0,
        "splits": {},
    }

    with tmp.open("w") as out:
        for folder, dirs, files in os.walk(destination):
            dirs.sort()
            if "videos" not in dirs and "cameras" not in dirs:
                continue

            scene = Path(folder)
            relative = scene.relative_to(destination)
            split = next(
                (
                    s for s in relative.parts
                    if s in ("train", "val", "test")
                ),
                "unknown",
            )
            videos = {
                p.stem: str(p.relative_to(ROOT))
                for p in sorted((scene / "videos").glob("*.mp4"))
            }
            camera = scene / "cameras/camera_extrinsics.json"
            problems = []
            camera_frame_count = None

            for cam in [f"cam{i:02d}" for i in range(1, 11)]:
                if (
                    cam not in videos
                    or (ROOT / videos[cam]).stat().st_size == 0
                ):
                    problems.append("missing_or_empty:" + cam)

            try:
                data = json.loads(camera.read_text())
                if not isinstance(data, (dict, list)) or not data:
                    problems.append("empty_camera_data")
                if split == 'train':
                    if not isinstance(data, dict):
                        problems.append('camera_frames_not_a_dict')
                    else:
                        frames = [key for key in data if re.fullmatch(r'frame\d+', key)]
                        camera_frame_count = len(frames)
                        expected_frames = {f'frame{i}' for i in range(81)}
                        if set(frames) != expected_frames:
                            problems.append('camera_frames_not_0_to_80')
                        invalid = [key for key in frames if not isinstance(data[key], dict) or any(
                            not isinstance(data[key].get(f'cam{i:02d}'), str) for i in range(1, 11))]
                        if invalid:
                            problems.append('missing_camera_pose_strings:' + ','.join(invalid[:5]))
            except (OSError, ValueError) as exc:
                problems.append("camera_json:" + str(exc))

            row = dict(
                dataset=name,
                scene_id=relative.as_posix(),
                split=split,
                scene_dir=str(scene.relative_to(ROOT)),
                videos=videos,
                extrinsics=str(camera.relative_to(ROOT)),
                structurally_complete=not problems,
                camera_frame_count=camera_frame_count,
                problems=problems,
            )
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["scenes"] += 1
            counts[
                "incomplete" if problems else "structurally_complete"
            ] += 1
            counts["splits"][split] = counts["splits"].get(split, 0) + 1

    tmp.replace(manifest)
    save(output / (name + "_summary.json"), counts)
    print(name, json.dumps(counts, ensure_ascii=False), flush=True)

    if counts["scenes"] == 0:
        raise RuntimeError(
            f"{name}: 未识别到场景，请检查实际目录结构"
        )
    return counts


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("已有数据准备任务运行，请查看日志")

        spec = importlib.util.spec_from_file_location(
            "prepare", ROOT / "tools/prepare_plenoptic.py"
        )
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)

        names = sys.argv[1:] or ["syncam", "multicam"]
        if any(n not in ("syncam", "multicam") for n in names):
            raise RuntimeError(
                "用法：prepare_datasets.py [syncam] [multicam]"
            )

        for name in names:
            repo, kind, revision, directory, filenames = prepare.JOBS[name]
            selected = sorted(
                f for f in filenames
                if re.search(
                    r"\.tar(?:\.(?:gz|zst))?$|\.part[a-z]+$", f
                )
            )
            if not selected:
                raise RuntimeError(
                    f"{name}: 下载清单中没有归档文件"
                )

            paths = [ROOT / directory / f for f in selected]
            inputs = []
            for filename, path in zip(selected, paths):
                key = hashlib.sha256(
                    f"{repo}/{revision}/{filename}".encode()
                ).hexdigest()
                receipt = (
                    ROOT / "download-state/verified" / (key + ".json")
                )
                if not path.is_file() or not receipt.is_file():
                    raise RuntimeError(
                        f"缺少文件或下载校验记录：{path}"
                    )
                if (
                    json.loads(receipt.read_text()).get("verified")
                    != fingerprint(path)
                ):
                    raise RuntimeError(
                        f"文件尚未校验或发生变化：{path}；"
                        "先运行 tools/download_weight_dataset.py start datasets"
                    )
                record = json.loads(receipt.read_text())
                inputs.append({'path': str(path.relative_to(ROOT)), 'size': record['size'], 'etag': record['etag']})

            print(
                f"开始 {name}，原包校验记录有效，输出盘可用 "
                f"{shutil.disk_usage(ROOT).free / 2**30:.1f} GiB",
                flush=True,
            )
            save(
                STATE / "status.json",
                dict(status="running", dataset=name, pid=os.getpid()),
            )
            counts = index(name, extract(name, paths, inputs))
            save(STATE / (name + ".done.json"), counts)

        save(
            STATE / "status.json",
            dict(status="completed", datasets=names),
        )
        print("全部解压和清单生成完成。", flush=True)


if __name__ == "__main__":
    main()
