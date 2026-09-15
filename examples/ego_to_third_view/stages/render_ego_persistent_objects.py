"""Reviewed contact intervals, persistent resting objects; isolated v6 ablation."""
from runtime import require_launcher
require_launcher()

import json
from pathlib import Path
import numpy as np
import render_ego_thirdview as base

original = base.object_tracks

def persistent_tracks(specs, n, *args, **kwargs):
    tracks, diagnostics = original(specs, n, *args, **kwargs)
    for spec in specs:
        raw = tracks[spec.key].copy()
        fixed = raw.copy()
        lo, hi = spec.held_range or (n, n - 1)
        static_spans = [(0, lo), (hi + 1, n)]
        report = []
        for start, stop in static_spans:
            if stop <= start:
                continue
            anchor = np.median(raw[start:stop], axis=0)
            anchor[2] = spec.size[2] / 2
            fixed[start:stop] = anchor
            report.append({"frames_half_open": [start, stop],
                           "raw_position_rms_m": float(np.sqrt(np.mean(np.sum((raw[start:stop]-anchor)**2, axis=1)))),
                           "output_position_rms_m": 0.0})
        # Correct only transition offsets with a smooth taper; retain the
        # observed moving interval and its timing, without smoothing over it.
        if hi >= lo and lo < n:
            width = min(12, (hi - lo + 1)//2)
            if lo > 0 and width > 0:
                delta = fixed[lo-1] - raw[lo]
                for j in range(width):
                    weight = 0.5 * (1 + np.cos(np.pi*j/max(1,width-1)))
                    fixed[lo+j] += weight*delta
            if hi+1 < n and width > 0:
                delta = fixed[hi+1] - raw[hi]
                for j in range(width):
                    weight = 0.5 * (1 + np.cos(np.pi*j/max(1,width-1)))
                    fixed[hi-j] += weight*delta
        tracks[spec.key] = fixed
        diagnostics[spec.key]["persistence_ablation"] = {
            "contact_source": "existing manually reviewed interval; not automatic contact detection",
            "static_spans": report,
            "max_correction_m": float(np.linalg.norm(fixed-raw,axis=1).max()),
            "note": "zero resting drift is imposed by model; it is not measured pose accuracy"}
    return tracks, diagnostics

if __name__ == "__main__":
    base.object_tracks = persistent_tracks
    base.main()
