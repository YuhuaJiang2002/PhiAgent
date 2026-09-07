#!/usr/bin/env python3
"""Retime only DiT using reviewed output/source frame correspondences."""
import argparse
from pathlib import Path

from automation.contracts import digest,read_json,write_json
from automation.media import probe,retime,same_clock
from automation.timeline import pchip_map


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dit',type=Path,required=True)
    parser.add_argument('--sim',type=Path,required=True)
    parser.add_argument('--anchors',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New output directory')
    args=parser.parse_args(argv)
    dit,sim=probe(args.dit),probe(args.sim)
    if not same_clock(dit,sim):raise ValueError('DiT and SIM must already share the delivery clock')
    config=read_json(args.anchors)
    if config['dit_sha256']!=digest(args.dit) or config['sim_sha256']!=digest(args.sim):
        raise ValueError('Anchor file belongs to different video versions')
    mapping=pchip_map(config['anchors'],dit['frames'])
    args.output.mkdir(parents=True,exist_ok=False)
    retime(args.dit,args.output/'dit_retimed.mp4',mapping)
    write_json(args.output/'mapping.json',{**mapping,'input_sha256':config['dit_sha256'],
        'sim_sha256':config['sim_sha256'],'output_sha256':digest(args.output/'dit_retimed.mp4'),
        'limitation':'Anchor interpolation is not independent timing validation; inspect full output and held-out motion.'})


if __name__=='__main__':main()
