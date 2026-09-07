"""FFmpeg media boundaries: original clock preservation, review sheets, frame selection."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import subprocess

from .contracts import digest, write_json
from .timeline import reference_frame_count


def probe(path):
    value=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-count_frames','-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration',
        '-of','json',str(path)],text=True))['streams'][0]
    fps=float(Fraction(value['avg_frame_rate']))
    frames=int(value['nb_read_frames'])
    if fps<=0 or frames<=0:
        raise ValueError('Invalid video clock')
    return {'width':int(value['width']),'height':int(value['height']),'fps':fps,
            'frames':frames,'duration_s':float(value.get('duration',frames/fps)),
            'reported_rate':value['r_frame_rate'],'average_rate':value['avg_frame_rate']}


def same_clock(left,right):
    return left['frames']==right['frames'] and abs(left['fps']-right['fps'])<1e-6 and abs(left['duration_s']-right['duration_s'])<1/left['fps']+.001


def run_ffmpeg(arguments):
    subprocess.run(['ffmpeg','-v','error','-n',*map(str,arguments)],check=True)


def prepare_reference(sim,out):
    """Only pad a separate conditioning copy. Never overwrite/retime the SIM authority."""
    spec=probe(sim)
    if abs(spec['fps']-24)>1e-6:
        raise ValueError('Pinned H3 adapter requires a reviewed 24fps control/export clock')
    count=reference_frame_count(spec['frames'])
    before=digest(sim)
    padding=count-spec['frames']
    run_ffmpeg(['-i',sim,'-vf',f'tpad=stop_mode=clone:stop={padding}',
        '-frames:v',count,'-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',
        '-movflags','+faststart',out])
    got=probe(out)
    if got['frames']!=count or digest(sim)!=before:
        raise RuntimeError('Reference padding contract failed')
    return {'authority_frames':spec['frames'],'reference_frames':count,'padding_frames':padding,
            'source_sha256':before,'reference_sha256':digest(out),
            'note':'17*n+5 input alignment avoids VAE tail truncation; not a hard action-timing guarantee'}


def retime(source,out,mapping):
    """Use decoded original frames only. No optical-flow hallucination or blends."""
    spec=probe(source)
    indices=mapping['source_frames']
    if len(indices)!=spec['frames'] or min(indices)<0 or max(indices)>=spec['frames']:
        raise ValueError('Mapping and video lengths differ')
    frame_bytes=spec['width']*spec['height']*3
    decoder=subprocess.Popen(['ffmpeg','-v','error','-i',str(source),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    encoder=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{spec["width"]}x{spec["height"]}','-r',str(spec['fps']),'-i','-',
        '-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    current=-1; frame=b''
    try:
        for index in indices:
            while current<index:
                chunks=[]; remaining=frame_bytes
                while remaining:
                    data=decoder.stdout.read(remaining)
                    if not data:raise RuntimeError('Missing decoded source frame')
                    chunks.append(data);remaining-=len(data)
                frame=b''.join(chunks); current+=1
            encoder.stdin.write(frame)
        encoder.stdin.close()
        # Consume to EOF to let FFmpeg exit normally, even if the contract changes.
        decoder.stdout.read()
        if decoder.wait()!=0 or encoder.wait()!=0:
            raise RuntimeError('Retiming codec failure')
    finally:
        for process in [decoder,encoder]:
            if process.poll() is None:
                process.terminate();process.wait()
        decoder.stdout.close()
        if not encoder.stdin.closed:encoder.stdin.close()
    if not same_clock(spec,probe(out)):
        raise RuntimeError('Retiming changed delivery clock')


def review_packet(videos,folder,events=None):
    """All-frame sheets, not first/middle/last-only review. No images are synthesized."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    result={'videos':{},'events':events or [],'coverage':'all frames; 24 thumbnails per sheet'}
    for label,path in videos.items():
        meta=probe(path)
        run_ffmpeg(['-i',path,'-vf',
            "scale=320:240:force_original_aspect_ratio=decrease,pad=320:260:(ow-iw)/2:20,drawtext=text='%{n}':x=4:y=2:fontsize=16:fontcolor=yellow,tile=4x6:nb_frames=24",
            '-fps_mode','vfr',folder/f'{label}_%04d.jpg'])
        result['videos'][label]={'path':str(Path(path).resolve()),'sha256':digest(path),**meta}
    write_json(folder/'packet.json',result)
    return result


def threeway(ego,sim,dit,out):
    if not same_clock(probe(sim),probe(dit)):
        raise ValueError('SIM/DiT clocks differ')
    if not same_clock(probe(ego),probe(sim)):
        raise ValueError('EGO/SIM clocks differ; use the reviewed source-clock export, not implicit fps conversion')
    filters=[]
    for i,name in enumerate(['EGO','SIM','DiT']):
        filters.append(f'[{i}:v]scale=768:576:force_original_aspect_ratio=decrease,pad=768:620:(ow-iw)/2:44,setsar=1,drawtext=text={name}:x=18:y=10:fontsize=24:fontcolor=white[v{i}]')
    filters.append('[v0][v1][v2]hstack=inputs=3[v]')
    run_ffmpeg(['-i',ego,'-i',sim,'-i',dit,'-filter_complex',';'.join(filters),'-map','[v]',
                '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',out])
