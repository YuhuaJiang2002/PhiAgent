#!/usr/bin/env python3
"""Publish latest validation videos and CPU-only convergence plots, without HTML."""
import plenoptic_paths as layout
import argparse
import csv
from collections import defaultdict
from datetime import datetime
import fcntl
from pathlib import Path
import socket
import tempfile

if __name__ == '__main__' and socket.gethostname().split('.')[0] != 'h20-1':
    raise RuntimeError('Validation plots may only be generated on h20-1')

import prepare_plenoptic as prepare
from validation_artifacts import (cleanup_legacy, commit_bundle, load_history,
                                  prepare_bundle, save)
from validation_plots import (clean, comparison_group, draw_dashboard, draw_view_control,
                              pyplot, read_training, save_figure, training_summary)

VALIDATION_ROOT = layout.rooted('outputs/validation')


def render_curves(training, history, output):
    groups = defaultdict(list)
    for record in history:
        groups[comparison_group(record)].append(record)
    latest = history[-1]
    current = comparison_group(latest)
    draw_dashboard(training, groups[current], output/'health-dashboard.png', history_count=len(history))
    draw_view_control(groups[current], output/'view-control.png')
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for key, records in groups.items():
        records = sorted(records, key=lambda r:(r['checkpoint_step'], r.get('created_at', '')))
        ax.plot([r['checkpoint_step'] for r in records], [r['target_mse'] for r in records],
                'o-', label=f'k={key[1]}, CP={key[2] if key[2] is not None else "unknown"}, suite={key[0][:8]}')
    ax.set(xlabel='Checkpoint step', ylabel='Target-only flow MSE',
           title='Fixed validation history: compare within each suite / k / CP')
    ax.legend(fontsize=9)
    save_figure(fig, output/'validation-loss.png')
    plt.close(fig)
    return dict(group_count=len(groups), dashboard_runs=len(groups[current]),
                dashboard_checkpoints=len({r['checkpoint_step'] for r in groups[current]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', help='Newly completed validation workspace')
    args = parser.parse_args()
    root = VALIDATION_ROOT
    root.mkdir(parents=True, exist_ok=True)
    run = (layout.rooted(args.run)).resolve() if args.run else None
    with (root/'.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        history, sources = load_history(root, run)
        with tempfile.TemporaryDirectory(prefix='.publish-', dir=root) as temporary:
            stage = Path(temporary)
            manifest = prepare_bundle(root, stage, history, sources)
            context = stage/'metrics/training_context'
            training = read_training(context/'metrics.jsonl', context/'config.json')
            training['source'] = str(root/'metrics/training_context/metrics.jsonl')
            curve_summary = render_curves(training, history, stage/'curves')
            save(stage/'metrics/diagnostics.json', clean(dict(
                updated_at=datetime.now().astimezone().isoformat(),
                checkpoint_step=manifest['checkpoint_step'],
                training=training_summary(training),
                validation_runs=len(history),
                **curve_summary,
                metric_definition='Target-only latent flow MSE; compare within the same suite/k/CP. '
                    'Training loss has different sampling and normalization. '
                    'Custom videos without a target reference do not contribute to validation loss.')))
            measured = []
            for case in history[-1]['cases']:
                if not case.get('image_metrics'):
                    continue
                metrics = case['image_metrics']
                measured.append(dict(step=history[-1]['checkpoint_step'], case=case['case_id'],
                    dataset=case['dataset'], scene=case['scene_id'], partition=case['partition'],
                    psnr_db=metrics['generated_vs_target']['psnr_db'],
                    copy_psnr_db=metrics['copy_source_vs_target']['psnr_db'],
                    psnr_gain_db=metrics['psnr_gain_over_copy_db'],
                    ssim=metrics['generated_vs_target']['luma_ssim'],
                    copy_ssim=metrics['copy_source_vs_target']['luma_ssim'],
                    generated_source_mae=metrics['generated_source_rgb_mae'],
                    target_source_mae=metrics['copy_source_vs_target']['rgb_mae']))
            if measured:
                save(stage/'metrics/image-errors.json', dict(cases=measured,
                    qualitative_cases_included=False, definition=metrics['definition']))
                with (stage/'metrics/image-errors.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(measured[0]))
                    writer.writeheader()
                    writer.writerows(measured)
            commit_bundle(stage, root)
        removed = cleanup_legacy(root)
    import json
    print(json.dumps(dict(checkpoint_step=manifest['checkpoint_step'],
        validation_runs=len(history), dashboard_runs=curve_summary['dashboard_runs'],
        video_cases=len(manifest['cases']),
        videos=str(layout.relative(root/'videos')),
        curves=str(layout.relative(root/'curves')), **removed)), flush=True)


if __name__ == '__main__':
    main()
