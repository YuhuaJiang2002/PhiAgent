"""Process-local Wan mask instrumentation used only by diagnosis runs."""

from __future__ import annotations

import utils

_original_get_aug_mask = utils.get_aug_mask
_frame_index = 0


def _diagnostic_get_aug_mask(body_mask, w_len=10, h_len=20):
    global _frame_index

    current = _frame_index
    _frame_index += 1
    if not body_mask.any():
        print(f"WAN_DIAG_EMPTY_MASK_FRAME={current}", flush=True)
        return body_mask
    return _original_get_aug_mask(body_mask, w_len=w_len, h_len=h_len)


utils.get_aug_mask = _diagnostic_get_aug_mask
