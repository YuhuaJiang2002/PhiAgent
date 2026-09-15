"""CPU-only plots for the canonical validation report."""
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np


def read_training(metrics, config=None, max_step=None):
    metrics = Path(metrics) if metrics else None
    result = dict(status='unavailable', rows=[], config={}, source=str(metrics or ''),
                  duplicate_steps=0, malformed_lines=0, nonfinite_steps=[], missing_steps=[])
    if config and Path(config).is_file():
        result['config'] = json.loads(Path(config).read_text())
    if not metrics or not metrics.is_file():
        return result
    latest = {}
    # Read a bounded snapshot even while the trainer appends to the file.
    with metrics.open('rb') as stream:
        raw = stream.read(metrics.stat().st_size)
    for line in raw.decode(errors='replace').splitlines():
        try:
            row = json.loads(line)
            if not {'step', 'loss', 'gradient_norm', 'seconds'} <= row.keys():
                continue
            step = int(row['step'])
            if max_step is not None and step > max_step:
                continue
            for key in ('loss', 'gradient_norm', 'seconds'):
                float(row[key])
        except (ValueError, TypeError, json.JSONDecodeError):
            result['malformed_lines'] += 1
            continue
        result['duplicate_steps'] += int(step in latest)
        latest[step] = row
    rows = [latest[step] for step in sorted(latest)]
    result['rows'] = rows
    if rows:
        result['status'] = 'available'
        result['nonfinite_steps'] = [r['step'] for r in rows if not all(
            math.isfinite(float(r[k])) for k in ('loss','gradient_norm','seconds'))]
        result['missing_steps'] = sorted(set(range(rows[0]['step'], rows[-1]['step']+1))-set(latest))
    return result


def phase(row):
    return (row.get('phase', 'unknown'), row.get('k', 'unknown'))


def comparison_group(record):
    return record['suite_sha256'], record['k'], record.get('context_parallel_size')


def smooth(values, rows, window, median=False, full=False):
    """Do not smooth across changes to the number of conditioning views."""
    result = np.full(len(values), np.nan)
    begin = 0
    for i, row in enumerate(rows):
        if i and phase(row) != phase(rows[i-1]):
            begin = i
        sample = values[max(begin, i-window+1):i+1]
        if full and (len(sample) < window or not np.isfinite(sample).all()):
            continue
        finite = sample[np.isfinite(sample)]
        if len(finite):
            result[i] = float(np.median(finite) if median else np.mean(finite))
    return result


def clean(value):
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [clean(v) for v in value]
    return value


def training_summary(training):
    rows = training['rows']
    result = {k:v for k,v in training.items() if k not in ('rows','config')}
    result.update(count=len(rows), windows=[], gradient_clip=training['config'].get('gradient_clip'),
                  learning_rate_from_config=training['config'].get('learning_rate'))
    if not rows:
        return result
    result.update(first_step=rows[0]['step'], last_step=rows[-1]['step'],
                  cutoff=rows[-1].get('updated_at'), current_k=rows[-1].get('k'))
    last_phase = phase(rows[-1])
    current = []
    for row in reversed(rows):
        if phase(row) != last_phase:
            break
        current.append(row)
    current.reverse()
    for label, selected in [('first_100', rows[:100]), ('current_k_last_500', current[-500:]),
                            ('current_k_last_100', current[-100:])]:
        result['windows'].append(dict(label=label, first_step=selected[0]['step'],
            last_step=selected[-1]['step'], count=len(selected),
            loss_mean=np.mean([r['loss'] for r in selected]),
            loss_median=np.median([r['loss'] for r in selected]),
            gradient_median=np.median([r['gradient_norm'] for r in selected]),
            gradient_max=np.max([r['gradient_norm'] for r in selected]),
            seconds_mean=np.mean([r['seconds'] for r in selected])))
    threshold = result['gradient_clip']
    if threshold is not None:
        result['clipped_current_k_last_100'] = sum(r['gradient_norm'] > threshold for r in current[-100:])
    return clean(result)


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.grid':True, 'grid.alpha':.18})
    return plt


def save_figure(fig, path, svg=False):
    path = Path(path)
    temporary = path.with_name(path.stem+'.tmp'+path.suffix)
    fig.savefig(temporary, dpi=145, facecolor='white')
    temporary.replace(path)
    if svg:
        target = path.with_suffix('.svg')
        temporary = target.with_name(target.stem+'.tmp.svg')
        fig.savefig(temporary, facecolor='white')
        temporary.replace(target)


def unavailable(ax, message):
    ax.text(.5, .5, message, ha='center', va='center', transform=ax.transAxes, color='#667085')


def draw_dashboard(training, records, output, *, history_count=None):
    """records must share the suite, number of conditions and CP size."""
    if len({comparison_group(r) for r in records}) > 1:
        raise ValueError('Cannot combine different validation protocols, k or CP in a dashboard')
    records = sorted(records, key=lambda r:(r['checkpoint_step'],r.get('created_at','')))
    plt = pyplot()
    fig, axes = plt.subplots(3,3,figsize=(18,12),constrained_layout=True)
    rows = training['rows']
    titles = ['Training loss (compare within k)', 'Recent training loss trend',
              'Gradient norm before clipping', 'Updates requiring clipping',
              'Step time (checkpoint writes excluded)', 'Memory peaks (not current usage)']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title)
        ax.set_xlabel('Optimizer step')
    if rows:
        xs = np.array([r['step'] for r in rows])
        loss = np.array([r['loss'] for r in rows],float)
        gradients = np.array([r['gradient_norm'] for r in rows],float)
        seconds = np.array([r['seconds'] for r in rows],float)
        for array in (loss,gradients,seconds):
            array[~np.isfinite(array)] = np.nan
        axes[0,0].plot(xs,loss,color='#94afd0',alpha=.5,lw=.55,label='raw')
        axes[0,0].plot(xs,smooth(loss,rows,100),color='#225c9f',lw=2,label='100-step mean')
        axes[0,0].legend(fontsize=9)
        axes[0,0].set_ylabel('Training-normalized flow MSE')
        for n,median,label,color in [(100,False,'100-step mean','#225c9f'),
                                      (500,False,'500-step mean','#d87d20'),
                                      (100,True,'100-step median','#328663')]:
            axes[0,1].plot(xs,smooth(loss,rows,n,median=median),label=label,color=color)
        # Autoscale only the visible recent range, without a hardcoded loss band.
        axes[0,1].set_xlim(max(xs[0]-.5,xs[-1]-1200),xs[-1]+.5)
        visible = [line.get_ydata()[xs >= max(xs[0],xs[-1]-1200)] for line in axes[0,1].lines]
        values = np.concatenate(visible)
        values = values[np.isfinite(values)]
        if len(values):
            padding=max(float(np.ptp(values))*.15,abs(float(np.mean(values)))*.03,1e-5)
            axes[0,1].set_ylim(float(np.min(values))-padding,float(np.max(values))+padding)
        axes[0,1].legend(fontsize=9)
        positive = np.where(gradients>0,gradients,np.nan)
        axes[0,2].plot(xs,positive,color='#a783bc',alpha=.5,lw=.6,label='pre-clip norm')
        axes[0,2].plot(xs,smooth(positive,rows,100,median=True),color='#6c348a',label='100-step median')
        threshold=training['config'].get('gradient_clip')
        if threshold is not None and threshold>0:
            axes[0,2].axhline(threshold,color='#c64949',ls='--',label=f'clip = {threshold:g}')
            events=np.where(np.isfinite(gradients),(gradients>threshold).astype(float)*100,np.nan)
            rate=smooth(events,rows,100,full=True)
            axes[1,0].plot(xs,rate,color='#c76b2d')
            axes[1,0].set_ylim(0,max(10,float(np.nanmax(rate)) if np.any(np.isfinite(rate)) else 10))
            axes[1,0].set_ylabel('Full trailing 100 updates (%)')
        else:
            unavailable(axes[1,0],'Clipping threshold unavailable')
        if np.any(np.isfinite(positive)):
            axes[0,2].set_yscale('log')
        axes[0,2].legend(fontsize=9)
        axes[1,1].plot(xs,seconds,color='#acb6bb',alpha=.4,lw=.6)
        axes[1,1].plot(xs,smooth(seconds,rows,100),color='#286675',label='100-step mean')
        axes[1,1].set_ylabel('Seconds')
        for key,label in [('max_rank_allocated_gib','allocated peak'),('max_rank_reserved_gib','reserved peak')]:
            axes[1,2].plot(xs,[r.get(key,np.nan) for r in rows],label=label)
        axes[1,2].set_ylabel('GiB')
        axes[1,2].legend(fontsize=9)
        for i,row in enumerate(rows[1:],1):
            if phase(row)!=phase(rows[i-1]):
                for ax in axes[:2].flat:
                    ax.axvline(row['step'],color='#7b8187',ls=':',lw=.8)
                axes[0,0].annotate(f'k={row.get("k","?")}',(row['step'],.97),
                    xycoords=('data','axes fraction'),ha='left',va='top',fontsize=8)
    else:
        for ax in axes[:2].flat:
            unavailable(ax,'Training logs unavailable on this host')
    axes[2,0].set_title('Fixed validation: target-only flow MSE')
    axes[2,1].set_title('Validation by noise level')
    axes[2,2].set_title('Validation by target camera')
    for ax in axes[2]:
        ax.set_xlabel('Checkpoint step')
    if records:
        xs=[r['checkpoint_step'] for r in records]
        levels=sorted({n['noise_level'] for r in records for c in r['cases'] for n in c['losses']})
        series={}
        for level in levels:
            means=[]
            for record in records:
                values=[n['target_mse'] for c in record['cases'] for n in c['losses'] if n['noise_level']==level]
                means.append(float(np.mean(values)) if values else np.nan)
            series[level]=np.asarray(means)
        if len(set(xs)) == 1:
            # One checkpoint has no trend; normalizing every noise level to 100
            # hides its measured value and draws all markers on top of each other.
            latest=records[-1]
            cases=sorted(latest['cases'],key=lambda c:c['case_id'])
            palette=['#2877ad','#dc8728','#328663','#be4d4d']
            panels=[
                ([str(xs[-1])],[latest['target_mse']],['#be4d4d']),
                ([f'{level:g}' for level in levels],[series[level][-1] for level in levels],palette),
                ([c['case_id'].replace('syncam-','syn-').replace('multicam-','multi-').replace('-cam','\ncam')
                  for c in cases],[c['target_mse'] for c in cases],palette),
            ]
            for ax,(labels,values,colors) in zip(axes[2],panels):
                positions=np.arange(len(values))
                bars=ax.bar(positions,values,width=.58,color=colors)
                ax.set_xticks(positions,labels,fontsize=9)
                ax.bar_label(bars,fmt='%.4f',padding=4,fontsize=9)
                ax.set_ylim(0,max(values)*1.2 if values and max(values)>0 else 1)
                ax.set_ylabel('Target-only flow MSE (lower is better)')
                ax.grid(axis='x',visible=False)
            axes[2,0].set_title('Fixed validation MSE: latest result')
            axes[2,1].set_title(f'Noise-level MSE at step {xs[-1]}')
            axes[2,1].set_xlabel('Noise level')
            axes[2,2].set_title(f'Target-camera MSE at step {xs[-1]}')
            axes[2,2].set_xlabel('Target camera')
        else:
            axes[2,0].plot(xs,[r['target_mse'] for r in records],'o-',color='#be4d4d')
            axes[2,0].set_ylabel('Lower is better')
            for record in records[-8:]:
                axes[2,0].annotate(f'{record["target_mse"]:.4f}',
                    (record['checkpoint_step'],record['target_mse']),xytext=(0,8),
                    textcoords='offset points',ha='center',fontsize=8)
            relative=bool(series) and all(np.isfinite(values[0]) and values[0]>0 for values in series.values())
            for level,values in series.items():
                plotted=values/values[0]*100 if relative else values
                axes[2,1].plot(xs,plotted,'o-',label=f'noise {level:g}')
            if relative:
                axes[2,1].axhline(100,color='#8d949d',ls='--',lw=.8)
                axes[2,1].set_title(f'Noise-level trend (step {xs[0]} = 100)')
            axes[2,1].set_ylabel('Relative MSE (%)' if relative else 'Target-only flow MSE')
            axes[2,1].legend(fontsize=9)
            for case in sorted({c['case_id'] for r in records for c in r['cases']}):
                means=[next((c['target_mse'] for c in r['cases'] if c['case_id']==case),np.nan) for r in records]
                axes[2,2].plot(xs,means,'o-',label=case.replace('syncam-',''))
            axes[2,2].set_ylabel('Target-only flow MSE')
            axes[2,2].legend(fontsize=9)
    else:
        for ax in axes[2]:
            unavailable(ax,'No completed validation for this group')
    training_label=(f'training through step {rows[-1]["step"]} at {rows[-1].get("updated_at","unknown time")}'
                    if rows else 'training logs unavailable')
    validation_label=(f'validation k={records[-1]["k"]}, CP={records[-1].get("context_parallel_size","unknown")}, suite={records[-1]["suite_sha256"][:8]}'
                      if records else 'no completed validation')
    checkpoints=len({r['checkpoint_step'] for r in records})
    count_label=(f'Current validation group: {len(records)} of {history_count if history_count is not None else len(records)} completed runs, '
                 f'{checkpoints} checkpoint(s).')
    if checkpoints == 1:
        count_label+=' Bottom panels show measured values; a trend needs another checkpoint in this group.'
    fig.suptitle(f'{training_label} | {validation_label}\n'
                 'Training and fixed validation losses have different sampling/normalization. Dashed vertical lines mark training stage changes.\n'
                 f'{count_label}',fontsize=12)
    save_figure(fig,output)
    plt.close(fig)


def draw_view_control(records, output):
    if len({comparison_group(r) for r in records}) > 1:
        raise ValueError('View-control trends require one fixed validation protocol')
    latest = records[-1]
    cases = [c for c in latest['cases'] if c.get('image_metrics')]
    if not cases:
        return
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    labels = [c['case_id'].replace('syncam-', 'syn-').replace('multicam-', 'multi-') for c in cases]
    xs = np.arange(len(cases))
    for ax, metric, title in [(axes[0,0], 'psnr_db', 'Agreement with true target: PSNR (higher is better)'),
                              (axes[0,1], 'luma_ssim', 'Agreement with true target: SSIM (higher is better)')]:
        actual = [c['image_metrics']['generated_vs_target'][metric] for c in cases]
        baseline = [c['image_metrics']['copy_source_vs_target'][metric] for c in cases]
        ax.bar(xs-.19, actual, .38, label='Generated target', color='#2877ad')
        ax.bar(xs+.19, baseline, .38, label='Copy input baseline', color='#a6afb8')
        ax.set_title(title)
        ax.set_xticks(xs, labels, rotation=30, ha='right', fontsize=8)
        ax.legend(fontsize=9)
    axes[0,0].set_ylabel('dB')
    axes[0,1].set_ylabel('Luminance SSIM')
    ax = axes[1,0]
    ax.bar(xs-.19, [c['image_metrics']['generated_source_rgb_mae'] for c in cases], .38,
           color='#c7842c', label='Generated vs input')
    ax.bar(xs+.19, [c['image_metrics']['copy_source_vs_target']['rgb_mae'] for c in cases], .38,
           color='#658b76', label='True target vs input')
    ax.set(title='Amount of visual change from input (not a correctness score)', ylabel='RGB MAE [0,1]')
    ax.set_xticks(xs, labels, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax = axes[1,1]
    for dataset, scene in sorted({(c['dataset'], c['scene_id']) for c in cases}):
        points = []
        for row in records:
            measured = [c['image_metrics']['psnr_gain_over_copy_db'] for c in row['cases']
                        if (c['dataset'], c['scene_id']) == (dataset, scene) and c.get('image_metrics')]
            if measured:
                points.append((row['checkpoint_step'], float(np.mean(measured))))
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], 'o-', label=dataset+' '+scene.split('/')[-1])
    ax.axhline(0, color='#777777', ls='--', lw=1)
    ax.set(title='Mean PSNR gain over copying the input', xlabel='Checkpoint step', ylabel='Gain (dB)')
    ax.legend(fontsize=9)
    fig.suptitle(f'Calibrated SynCam / MultiCam | step {latest["checkpoint_step"]}, k={latest["k"]}, CP=4\n'
                 'Same source and seed across target cameras. Robot visualization is excluded from every error metric.', fontsize=12)
    save_figure(fig, output)
    plt.close(fig)


def decode_frame(path,index):
    from PIL import Image
    if not shutil.which('ffmpeg'):
        # Local offline previews may have PyAV but no ffmpeg executable.
        import av
        with av.open(str(path)) as container:
            for i,frame in enumerate(container.decode(video=0)):
                if i==index:
                    return frame.to_image().convert('RGB')
        raise ValueError(f'Cannot decode frame {index}: {path}')
    command=['ffmpeg','-v','error','-threads','1','-i',str(path),'-vf',f'select=eq(n\\,{index})',
             '-frames:v','1','-threads','1','-f','image2pipe','-vcodec','png','pipe:1']
    result=subprocess.run(command,capture_output=True,timeout=45,check=True)
    return Image.open(BytesIO(result.stdout)).convert('RGB')


def draw_comparison(records,output,limit=4):
    """A bounded visual comparison; rendered references never become metric inputs."""
    from PIL import Image
    selected=[]
    for record in sorted(records,key=lambda r:r.get('created_at','')):
        if record.get('loss_only') or not record['cases']:
            continue
        if all(c.get('video_directory') and (Path(record['_run'])/c['video_directory']/'generated.mp4').is_file()
               for c in record['cases']):
            selected.append(record)
    selected=selected[-limit:]
    if not selected:
        return dict(status='unavailable',reason='No completed generated videos in this suite/k group')
    plt=pyplot()
    cases=selected[-1]['cases']
    fig,axes=plt.subplots(len(cases),len(selected)+1,
        figsize=(4*(len(selected)+1),2.8*len(cases)),squeeze=False)
    warnings=[]
    compared=[]
    def picture(record,case_id,reference=False):
        case=next(c for c in record['cases'] if c['case_id']==case_id)
        folder=Path(record['_run'])/case['video_directory']
        info=json.loads((folder/'inference.json').read_text())
        middle=int(info['frames'])//2
        comparison=json.loads((folder/'comparison.json').read_text()) if (folder/'comparison.json').is_file() else {}
        key='target_reference' if reference else 'generated'
        saved=comparison.get('preview_images',{}).get(key)
        if saved and comparison.get('preview_frame_index')==middle and (folder/saved).is_file():
            return Image.open(folder/saved).convert('RGB'),middle,False
        if not reference:
            return decode_frame(folder/'generated.mp4',middle),middle,False
        image=decode_frame(folder/'comparison.mp4',middle)
        panel_width=image.width//3
        panel_height=round(panel_width*info['height']/info['width']/2)*2
        return image.crop((panel_width*2,image.height-panel_height,image.width,image.height)),middle,True
    for i,case in enumerate(cases):
        name=case['case_id']
        for j,record in enumerate(selected):
            try:
                image,index,_=picture(record,name)
                axes[i,j].imshow(image)
                compared.append(dict(step=record['checkpoint_step'],case_id=name,frame_index=index))
            except (OSError,ValueError,KeyError,StopIteration,subprocess.SubprocessError) as error:
                warnings.append(f'{name}, step {record["checkpoint_step"]}: {type(error).__name__}')
                unavailable(axes[i,j],'Preview unavailable')
            axes[i,j].set_title(f'{name} | step {record["checkpoint_step"]}',fontsize=9)
            axes[i,j].axis('off')
        rendered=True
        try:
            image,index,rendered=picture(selected[-1],name,reference=True)
            axes[i,-1].imshow(image)
        except (OSError,ValueError,KeyError,StopIteration,subprocess.SubprocessError) as error:
            warnings.append(f'{name}, reference: {type(error).__name__}')
            unavailable(axes[i,-1],'Reference unavailable')
        axes[i,-1].set_title(f'{name} | reference'+(' (rendered)' if rendered else ''),fontsize=9)
        axes[i,-1].axis('off')
    fig.suptitle('Same middle frame across the latest four video validations in this suite/k group\n'
                 'Compare camera, subject scale and identity. Watch the videos to assess temporal consistency.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94),h_pad=1.5)
    save_figure(fig,output)
    plt.close(fig)
    return dict(status='rendered',checkpoints=[r['checkpoint_step'] for r in selected],
                compared=compared,warnings=warnings)
