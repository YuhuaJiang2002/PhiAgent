#!/usr/bin/env python3
"""Download only the pinned external Sol adapter/kernel subset; no weights/install."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION='6fb7eb11c3435555ec6d6adf0d5572d339d2c6eb'
PREFIXES=('models/minimax_h3/A100/','techniques/sparse_backends/sol_attn/')


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'PhiAgent-pinned-source-preparer'})
    with urllib.request.urlopen(request,timeout=60) as response:return response.read()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New external source directory, outside repository')
    parser.add_argument('--existing',type=Path,help='Verify/copy a previously downloaded subset against the pinned Git tree')
    args=parser.parse_args(argv)
    destination=args.output.resolve()
    repository=Path(__file__).resolve().parents[2]
    if destination==repository or repository in destination.parents:
        parser.error('Third-party sources must stay outside the repository')
    tree=json.loads(fetch(f'https://api.github.com/repos/NVlabs/Sana/git/trees/{REVISION}?recursive=1'))
    if tree.get('sha')!=REVISION or tree.get('truncated'):raise RuntimeError('Unexpected/incomplete pinned tree')
    files=[r for r in tree['tree'] if r['type']=='blob' and (r['path'].startswith(PREFIXES) or r['path'] in ('AGENTS.md','LICENSE'))]
    if not files:raise RuntimeError('Pinned source subset is empty')
    destination.mkdir(parents=True,exist_ok=False)
    hashes={}
    for entry in files:
        relative=Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe upstream path')
        content=(args.existing/relative).read_bytes() if args.existing and (args.existing/relative).is_file() else fetch(f'https://raw.githubusercontent.com/NVlabs/Sana/{REVISION}/{entry["path"]}')
        blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if blob!=entry['sha']:raise RuntimeError(f'Git blob verification failed: {relative}')
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        hashes[str(relative)]=hashlib.sha256(content).hexdigest()
    (destination/'phiagent_source_manifest.json').write_text(json.dumps({'revision':REVISION,'sha256':hashes},indent=2)+'\n')
    print(json.dumps({'verified_files':len(hashes),'output':str(destination),'revision':REVISION}))


if __name__=='__main__':main()
