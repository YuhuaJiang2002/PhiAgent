from __future__ import annotations

import pickle

import numpy as np
import pytest

from scripts.run_cosmos_predict2_droid_inference import (
    extract_lora_state_dict,
    install_cpu_text_encoding,
    install_precomputed_text_encoding,
    load_t5_embedding,
    lora_target_modules,
    scale_lora_state_dict,
    validate_gpu_selection,
)


INVENTORY = [
    {
        "physical_index": index,
        "uuid": f"GPU-{index}",
        "name": "A800",
        "memory_total_mib": 81920,
        "memory_used_mib": used,
        "memory_free_mib": 81920 - used,
        "utilization_gpu_percent": 0,
    }
    for index, used in enumerate((5, 10_000, 50_000, 5))
]


def test_gpu_selection_preserves_explicit_physical_order() -> None:
    rows = validate_gpu_selection(INVENTORY, [3, 0], 70_000)
    assert [row["physical_index"] for row in rows] == [3, 0]


def test_gpu_selection_fails_closed_on_insufficient_memory() -> None:
    with pytest.raises(RuntimeError, match="below 70000 MiB"):
        validate_gpu_selection(INVENTORY, [1, 2], 70_000)


def test_gpu_selection_rejects_invalid_context_parallel_count() -> None:
    with pytest.raises(ValueError, match="1, 2, 4, or 8"):
        validate_gpu_selection(INVENTORY, [0, 1, 3], 1)


def test_lora_checkpoint_filter_keeps_only_prefixed_adapter_tensors() -> None:
    raw = {
        "net.blocks.0.q_proj.lora_A.default.weight": "a",
        "net.blocks.0.q_proj.lora_B.default.weight": "b",
        "net.blocks.0.q_proj.weight": "base",
        "optim.foo.lora_A.default.weight": "optim",
    }
    assert extract_lora_state_dict(raw) == {
        "blocks.0.q_proj.lora_A.default.weight": "a",
        "blocks.0.q_proj.lora_B.default.weight": "b",
    }


def test_attention_lora_profile_excludes_memory_heavy_mlp_branches() -> None:
    assert lora_target_modules("attention") == [
        "q_proj",
        "k_proj",
        "v_proj",
        "output_proj",
    ]
    assert "mlp.layer1" in lora_target_modules("full")


def test_lora_residual_scale_is_applied_to_b_projection_exactly_once() -> None:
    state = {
        "blocks.0.q_proj.lora_A.default.weight": 2.0,
        "blocks.0.q_proj.lora_B.default.weight": 4.0,
    }
    assert scale_lora_state_dict(state, 0.25) == {
        "blocks.0.q_proj.lora_A.default.weight": 2.0,
        "blocks.0.q_proj.lora_B.default.weight": 1.0,
    }
    with pytest.raises(ValueError, match="LoRA scale"):
        scale_lora_state_dict(state, 0.0)


def test_cpu_text_encoding_keeps_token_ids_and_weights_on_cpu() -> None:
    class TextEncoder:
        device = "cuda"

        def __init__(self) -> None:
            self.moves = []

        def to(self, *, device):
            self.moves.append(device)
            return self

        def encode_prompts(self, prompts, *, max_length, return_mask):
            return (prompts, self.device, max_length, return_mask)

    class Pipe:
        text_encoder = TextEncoder()

    pipe = Pipe()
    install_cpu_text_encoding(pipe)
    assert pipe.encode_prompt("task", max_length=12) == ("task", "cpu", 12, False)
    assert pipe.text_encoder.moves == ["cpu"]


def test_precomputed_text_encoding_is_prompt_exact() -> None:
    class Pipe:
        text_encoder = None

    pipe = Pipe()
    install_precomputed_text_encoding(
        pipe,
        prompt="positive",
        negative_prompt="negative",
        prompt_embedding="positive tensor",
        negative_prompt_embedding="negative tensor",
    )
    assert pipe.encode_prompt("positive") == "positive tensor"
    assert pipe.encode_prompt("negative") == "negative tensor"
    with pytest.raises(ValueError, match="does not match"):
        pipe.encode_prompt("different")


def test_load_t5_embedding_fails_closed_on_wrong_width(tmp_path) -> None:
    path = tmp_path / "embedding.pickle"
    with path.open("wb") as handle:
        pickle.dump([np.zeros((4, 768), dtype=np.float16)], handle)
    with pytest.raises(ValueError, match="invalid Cosmos T5"):
        load_t5_embedding(path)
