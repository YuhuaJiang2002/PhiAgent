#!/usr/bin/env python3
"""Scene-independent batch automation; dry-run by default, explicit execution/resume."""
import argparse
import json
from pathlib import Path

from automation.contracts import INVARIANTS,read_json,validate_manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true',help='Run trusted local adapters and configured model requests')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    manifest=validate_manifest(read_json(args.manifest))
    # All paths in a batch are explicit absolute paths; no cwd-dependent scenes.
    for clip in manifest['clips']:
        for key in ['source','bundle','appearance','existing_dit']:
            if key in clip and not Path(clip[key]).is_absolute():
                parser.error(f'{clip["id"]}.{key} must be an absolute path')
    if not args.execute:
        print(json.dumps({'mode':'plan_only','clips':[c['id'] for c in manifest['clips']],
            'pipeline':['reconstruct','stabilize','render','visual_precheck','resident_h3','visual_postcheck','bounded_repair','deliver'],
            'invariants':INVARIANTS,'note':'No GPU jobs or model downloads started. Raw RGB needs a configured perception adapter.'},indent=2))
        return 0
    from automation.factory import Factory
    result=Factory(manifest,args.output,resume=args.resume).run()
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0 if all(c['status']=='COMPLETE' for c in result.values()) else 2


if __name__=='__main__':raise SystemExit(main())
