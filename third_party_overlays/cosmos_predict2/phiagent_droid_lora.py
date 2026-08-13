"""Hydra experiment overlay for PhiAgent DROID multiview LoRA training.

This file is copied into the pinned Cosmos Predict2 checkout at run time.  It
keeps all heavyweight imports in the optional third-party environment.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.configs.base.config_video2world import (
    get_cosmos_predict2_video2world_pipeline,
)
from cosmos_predict2.data.dataset_video import Dataset
from cosmos_predict2.models.video2world_model import (
    Predict2ModelManagerConfig,
    Predict2Video2WorldModel,
    Predict2Video2WorldModelConfig,
)
from imaginaire.lazy_config import LazyCall as L


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is absent: {name}")
    return value


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _bounded_env_float(name: str, default: float, maximum: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not 0.0 < value <= maximum:
        raise ValueError(f"{name} must be in (0, {maximum}]")
    return value


def get_sampler(dataset: Dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=20260812,
    )


class PhiAgentDroidLoRAModel(Predict2Video2WorldModel):
    """Save adapter tensors only, avoiding 28 GB base duplication per checkpoint."""

    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: Any,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prioritize both exterior tiles and downweight the inactive black tile."""
        output, loss, mse, edm = super().compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W,
            condition,
            epsilon_B_C_T_H_W,
            sigma_B_T,
        )
        height, width = loss.shape[-2:]
        if height % 2 or width % 2:
            raise RuntimeError(f"native 2x2 latent grid must be even, got {height}x{width}")
        weights = torch.empty(
            (1, 1, 1, height, width), device=loss.device, dtype=loss.dtype
        )
        weights[..., : height // 2, :] = 1.4
        weights[..., height // 2 :, : width // 2] = 0.8
        weights[..., height // 2 :, width // 2 :] = 0.1
        weights = weights / weights.mean()
        weighted_loss = loss * weights
        output["loss"] = weighted_loss.mean()
        output["view_weighted_edm_loss"] = weighted_loss.mean()
        return output, weighted_loss, mse, edm

    def state_dict(self) -> dict[str, Any]:
        state = {
            key: value
            for key, value in self.pipe.dit.state_dict(prefix="net.").items()
            if ".lora_A." in key or ".lora_B." in key
        }
        if not state:
            raise RuntimeError("LoRA-only checkpoint contains no adapter tensors")
        for key, value in state.items():
            state[key] = (
                value.full_tensor().detach().cpu()
                if isinstance(value, DTensor)
                else value.detach().cpu()
            )
        return state


DATASET_DIR = _required_env("PHIAGENT_DROID_TRAIN_DATASET")
BASE_CHECKPOINT = _required_env("PHIAGENT_DROID_BASE_CHECKPOINT")
TOKENIZER = _required_env("PHIAGENT_DROID_TOKENIZER")
RUN_NAME = _required_env("PHIAGENT_RUN_NAME")
MAX_ITER = _positive_env_int("PHIAGENT_MAX_ITER", 1500)
SAVE_ITER = _positive_env_int("PHIAGENT_SAVE_ITER", 300)
WARMUP_ITER = min(20, max(1, MAX_ITER // 10))
GPU_COUNT = _positive_env_int("PHIAGENT_GPU_COUNT", 8)
if GPU_COUNT not in (4, 8):
    raise ValueError("PHIAGENT_GPU_COUNT must be 4 or 8")
LORA_RANK = _positive_env_int("PHIAGENT_LORA_RANK", 16)
if LORA_RANK not in (8, 16, 32):
    raise ValueError("PHIAGENT_LORA_RANK must be 8, 16, or 32")
LEARNING_RATE = _bounded_env_float("PHIAGENT_LEARNING_RATE", 1e-4, 1e-3)
TRAIN_FRAMES = _positive_env_int("PHIAGENT_TRAIN_FRAMES", 45)
if TRAIN_FRAMES not in (29, 45, 61, 77, 93):
    raise ValueError("PHIAGENT_TRAIN_FRAMES must be CP4-compatible: 29, 45, 61, 77, or 93")
TRAIN_STATE_T = (TRAIN_FRAMES - 1) // 4 + 1
if TRAIN_STATE_T % GPU_COUNT:
    raise ValueError("latent temporal length must divide the context-parallel GPU count")

pipe_config = get_cosmos_predict2_video2world_pipeline(
    model_size="14B", resolution="480", fps=16
)
pipe_config.ema.enabled = False
pipe_config.state_t = TRAIN_STATE_T
pipe_config.prompt_refiner_config.enabled = False
pipe_config.guardrail_config.enabled = False
pipe_config.min_num_conditional_frames = 1
pipe_config.max_num_conditional_frames = 1
pipe_config.tokenizer.vae_pth = TOKENIZER
# Captions are pre-encoded.  Loading T5 once per rank during training is both
# unnecessary and a major memory regression in the upstream training path.
pipe_config.text_encoder.t5.ckpt_path = ""

dataset = L(Dataset)(
    dataset_dir=DATASET_DIR,
    num_frames=TRAIN_FRAMES,
    video_size=(432, 768),
)
dataloader = L(DataLoader)(
    dataset=dataset,
    sampler=L(get_sampler)(dataset=dataset),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

experiment = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_14b_480p_16fps"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        "_self_",
    ],
    model=L(PhiAgentDroidLoRAModel)(
        config=Predict2Video2WorldModelConfig(
            train_architecture="lora",
            lora_rank=LORA_RANK,
            lora_alpha=LORA_RANK,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj",
            init_lora_weights=True,
            pipe_config=pipe_config,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path=BASE_CHECKPOINT,
                text_encoder_path="",
            ),
            fsdp_shard_size=GPU_COUNT,
            high_sigma_ratio=0.05,
        ),
        _recursive_=False,
    ),
    model_parallel=dict(context_parallel_size=GPU_COUNT),
    dataloader_train=dataloader,
    trainer=dict(
        distributed_parallelism="fsdp",
        seed=20260812,
        max_iter=MAX_ITER,
        logging_iter=1,
        run_validation=False,
        callbacks=dict(iter_speed=dict(hit_thres=10)),
    ),
    checkpoint=dict(save_iter=SAVE_ITER),
    optimizer=dict(lr=LEARNING_RATE, weight_decay=0.01),
    scheduler=dict(
        warm_up_steps=[WARMUP_ITER],
        cycle_lengths=[MAX_ITER],
        f_max=[1.0],
        f_min=[0.05],
    ),
    job=dict(
        project="phiagent",
        group="droid_multiview_lora_14b",
        name=RUN_NAME,
    ),
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="phiagent_droid_lora_attention_14b_480p_16fps",
    node=experiment,
)
