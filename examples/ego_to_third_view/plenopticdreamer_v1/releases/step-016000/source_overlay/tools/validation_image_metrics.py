"""Pixel errors on decoded full clips, including a copy-the-input baseline."""
import math
import numpy as np


def _luma_ssim(first, second):
    import cv2
    weights = np.array([.299, .587, .114], dtype=np.float32)
    x, y = first @ weights, second @ weights
    def blur(value):
        return cv2.GaussianBlur(value, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    vx = np.maximum(blur(x*x) - mx*mx, 0.)
    vy = np.maximum(blur(y*y) - my*my, 0.)
    covariance = blur(x*y) - mx*my
    score = ((2*mx*my + .01**2) * (2*covariance + .03**2)
             / ((mx*mx + my*my + .01**2) * (vx + vy + .03**2)))
    return float(score[5:-5, 5:-5].mean(dtype=np.float64))


def pair_metrics(first, second):
    if (first.shape != second.shape or first.ndim != 4 or first.shape[-1] != 3
            or first.dtype != np.uint8 or second.dtype != np.uint8
            or min(first.shape[1:3]) < 11 or len(first) == 0):
        raise ValueError('Metrics require matching RGB uint8 T,H,W,3 clips')
    mse = mae = ssim = 0.
    for a, b in zip(first, second):
        x, y = a.astype(np.float32)/255., b.astype(np.float32)/255.
        diff = x-y
        mse += float(np.square(diff).mean(dtype=np.float64))
        mae += float(np.abs(diff).mean(dtype=np.float64))
        ssim += _luma_ssim(x, y)
    mse, mae, ssim = mse/len(first), mae/len(first), ssim/len(first)
    return dict(rgb_mse=mse, rgb_mae=mae,
        psnr_db=-10*math.log10(max(mse, 1e-10)), psnr_capped_at_100_db=mse < 1e-10,
        luma_ssim=ssim)


def measure_video_errors(generated, reference, source):
    quality = pair_metrics(generated, reference)
    baseline = pair_metrics(source, reference)
    # This distance alone does not prove correct view synthesis; quality against
    # the true target determines whether moving away from the source helped.
    generated_source_mae = sum(float(np.abs(a.astype(np.float32)-b.astype(np.float32)).mean(
        dtype=np.float64))/255. for a, b in zip(generated, source))/len(generated)
    return dict(schema=1, frames=len(generated), height=generated.shape[1], width=generated.shape[2],
        generated_vs_target=quality, copy_source_vs_target=baseline,
        psnr_gain_over_copy_db=quality['psnr_db']-baseline['psnr_db'],
        ssim_gain_over_copy=quality['luma_ssim']-baseline['luma_ssim'],
        generated_source_rgb_mae=generated_source_mae,
        definition=f'All aligned frames at {generated.shape[1]}x{generated.shape[2]}. RGB MSE/MAE use [0,1]; PSNR uses clip-mean RGB MSE, capped at 100 dB. '
                   'SSIM uses BT.601 luminance, an 11x11 Gaussian window (sigma=1.5), population covariance and valid pixels. '
                   'The generated MP4 is decoded after encoding. The baseline copies the first conditioning camera. '
                   'Pixel errors measure agreement with the target; they are not estimated camera rotation/translation errors.')
