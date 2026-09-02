# CacheVDA RGB-to-relative-depth integration

Status: **WORKING** for isolated MP4-to-relative-depth-visualization inference.

CacheVDA-B-FP16 is integrated as an optional subprocess backend. PhiAgent does
not import or install CacheVDA's PyTorch, CUDA, xFormers, OpenCV, or NumPy
dependencies.

```text
RGB MP4
  -> PhiAgent lightweight adapter and physical-GPU audit
  -> /data1/zhn/Video-Depth-Anything-FeatureCache/.venv
  -> CacheVDA-B-FP16
  -> relative-depth visualization MP4 + timing JSON
```

## Environment boundary

The two environments remain independent:

```text
/data1/zhn/PhiAgent/.venv
  lightweight adapter, logging, hashes, GPU selection, output validation

/data1/zhn/Video-Depth-Anything-FeatureCache/.venv
  Python 3.10.19, PyTorch 2.12.1+cu130, torchvision 0.27.1+cu130,
  xFormers 0.0.35, OpenCV 4.11.0, NumPy 1.24.0
```

Do not install the external repository's requirements into PhiAgent.

## Semantics and claim boundary

- The model is official relative-depth VDA-Base (`vitb`) with CacheVDA's exact
  DINOv2 history-feature reuse and CUDA FP16 autocast.
- The result is affine-ambiguous relative depth, not depth in metres.
- The current PhiAgent backend saves an Inferno/gray H.264 visualization. The
  visualization is globally normalized, 8-bit, and compressed; it must not be
  used as numeric depth for geometry, calibration, collision, or evaluation.
- Raw float32 relative-depth integration is not yet exposed by the PhiAgent
  subprocess contract. It remains **NOT STARTED** here even though CacheVDA has
  an in-process experimental function and a BONN `.npy` evaluation entrypoint.
- The VDA-Base checkpoint is CC-BY-NC-4.0. The integration is for local research
  reproduction unless the use case separately satisfies the model license.

## Pinned local assets

PhiAgent verifies all of the following before every run:

- upstream base commit
  `4f5ae23172ba60fd7bc11ef671cca678842c7072`;
- three CacheVDA core-file SHA-256 values recorded in
  `phiagent/perception/cachevda.py`;
- checkpoint size `458247082` bytes and SHA-256
  `775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b`;
- CUDA availability and FP16 autocast in the external `.venv`;
- requested FFmpeg encoder and minimum free GPU memory.

The CacheVDA experiment files are still uncommitted in their source repository.
Exact file hashes make this machine-local integration auditable, but portable
Git reproduction remains **PARTIAL** until those files are committed and tagged.

## PhiAgent entrypoints

- Adapter: `phiagent/perception/cachevda.py`
- Audited CLI: `scripts/run_cachevda_depth.py`
- CPU tests: `tests/test_cachevda_adapter.py`

Example:

```bash
.venv/bin/python scripts/run_cachevda_depth.py \
  --repository /data1/zhn/Video-Depth-Anything-FeatureCache \
  --input-video /absolute/path/rgb.mp4 \
  --experiment-dir outputs/cachevda-rgb-depth/UNIQUE-RUN \
  --gpu 1 \
  --minimum-free-gpu-mib 12288 \
  --max-frames 120 \
  --input-size 518 \
  --max-res 1280 \
  --warmup-windows 1 \
  --preprocess-workers 8 \
  --encode-batch-size 32 \
  --encoder h264_nvenc \
  --log-every 1
```

Each invocation refuses an existing experiment directory and records:

- physical GPU inventory, process inventory, selection, and lease;
- PhiAgent Git state, CacheVDA base commit, exact core-file hashes, checkpoint
  hash, host, platform, command, and external package freeze;
- subprocess log, native CacheVDA timing JSON, decoded-video probe, output hash,
  status, and claim boundary.

Success requires all of the following:

1. subprocess return code is zero;
2. native timing JSON has `status == "completed"`;
3. timing JSON reports positive inference windows and CUDA memory use;
4. output contains exactly one decodable video stream and no audio;
5. decoded frame count equals the native timing JSON frame count;
6. requested `--max-frames`, when positive, equals the produced frame count.

## First PhiAgent acceptance run

Run directory:

```text
outputs/cachevda-rgb-depth/20260902T022500Z-bwm-demo-episode40-120f
```

The input was frames 0-119 of BWM's released RoboTwin episode 40 RGB video.

| Field | Observed |
| --- | ---: |
| Status | WORKING |
| Frames | 120 |
| Resolution | 640x480 |
| FPS / duration | 30 / 4.0 s |
| Windows | 6 |
| Model load | 1.799 s |
| Inference + online alignment | 3.203 s |
| End-to-end before metrics write | 7.617 s |
| CUDA peak allocated | 4.307 GiB |
| CUDA peak reserved | 5.508 GiB |
| Output SHA-256 | `8dd868b7e06c367af3d767181633c006e3399b95fd9b7f32f1b35b88ff8fcd85` |

The output fully decodes as 120 H.264 frames. A frames 0/60/119 review shows
coherent foreground/background and robot/bottle relative-depth structure. This
is a runtime and integration acceptance only; it is not a metric-depth,
held-out-quality, or physical-robot result. Because this input is 640x480 and
the GPU was shared, its timing must not replace CacheVDA's formal 1280x720
performance baseline.

## Next boundary

If PhiAgent needs depth for 3-D geometry rather than visualization, define a
separate versioned subprocess output contract for float32 arrays, including:

- `[T,H,W]` shape, dtype, finite/range checks, frame timestamps, and RGB lineage;
- explicit relative-depth semantics and optional independent metric anchors;
- chunked storage or bounded-memory output for long videos;
- exact array hashes and a consumer-side prohibition on treating values as
  metres before calibration.
