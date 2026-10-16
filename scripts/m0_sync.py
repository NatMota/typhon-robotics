#!/usr/bin/env python3
"""Reproduce M0 camera pairing evidence, audio offsets and unrotated previews.

Run on simulation host: .venv-sim/bin/python scripts/m0_sync.py
Candidate pairs and visual event anchors are reviewed annotations, not learned
matching. All 16 possible cross-camera audio matches are evaluated independently.
Offset convention: secondary_time = primary_time + offset_s.
"""
import csv
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/personal/m0'
RESULTS = ROOT / 'results/personal-m0-20261009'
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
RATE = 100  # 10 ms envelope bins; this is not independently calibrated A/V latency.
PAIRS = [
    (5, 1, [9, 12, 19, 29, 36, 48, 55, 64, 78, 84, 91]),
    (7, 2, [5, 6, 7, 12, 14, 19, 20, 21, 26, 27, 32, 33]),
    (8, 3, [4, 6, 8, 15, 21, 24, 29, 37, 39, 44, 50, 53, 57, 63, 65, 70, 76, 79]),
    (6, 4, []),
]


def overlap_score(primary, secondary, lag, start=None, end=None):
    ip = max(0, -lag) if start is None else start
    js = ip + lag
    n = min(len(primary) - ip, len(secondary) - js)
    if end is not None:
        n = min(n, end - ip)
    if ip < 0 or js < 0 or n <= 0:
        return None
    x, y = primary[ip:ip + n], secondary[js:js + n]
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt((x @ x) * (y @ y))
    return (float(x @ y / den) if den > 0 else 0., n)


def best_offsets(primary, secondary, lags, min_overlap, start=None, end=None):
    scores = []
    for lag in lags:
        result = overlap_score(primary, secondary, lag, start, end)
        if result is not None and result[1] >= min_overlap:
            scores.append((result[0], lag / RATE, result[1] / RATE))
    scores.sort(reverse=True)
    peaks = []
    for corr, offset, overlap in scores:
        if all(abs(offset - p['offset_s']) > .5 for p in peaks):
            peaks.append(dict(correlation=corr, offset_s=offset, overlap_s=overlap))
        if len(peaks) == 3:
            break
    return peaks


def audio_envelope(source):
    samples = np.frombuffer(subprocess.check_output([
        FFMPEG, '-v', 'error', '-i', str(source), '-vn', '-ac', '1',
        '-ar', '8000', '-f', 'f32le', 'pipe:1']), dtype=np.float32)
    usable = len(samples) // 80 * 80
    frames = samples[:usable].reshape(-1, 80)
    return np.log(np.sqrt((frames * frames).mean(axis=1)) + 1e-5)


def preview(primary, secondary, row):
    start = max(0., -row['offset_s']) + 2
    duration, fps = 12., 15.
    caps = [cv2.VideoCapture(str(primary)), cv2.VideoCapture(str(secondary))]
    intermediate = RESULTS / 'sync_intermediate' / (row['pair_id'] + '_preview_mjpeg.avi')
    output = RESULTS / 'synchronization_previews' / (row['pair_id'] + '.mp4')
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'MJPG'), fps, (640, 620))
    if not writer.isOpened():
        raise RuntimeError('Cannot create synchronization preview')
    try:
        for frame in range(round(duration * fps)):
            pt = start + frame / fps
            times = [pt, pt + row['offset_s']]
            canvas = np.full((620, 640, 3), 235, np.uint8)
            for i, (cap, t) in enumerate(zip(caps, times)):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"Preview decode failed: {row['pair_id']} camera{i} t={t}")
                h, w = image.shape[:2]
                scale = min(320 / w, 565 / h)
                image = cv2.resize(image, (round(w * scale), round(h * scale)))
                x, y = i * 320 + (320 - image.shape[1]) // 2, 50
                canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
                label = f"{'Primary' if i == 0 else 'Secondary'} t={t:.2f}s"
                cv2.putText(canvas, label, (i * 320 + 6, 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
            cv2.putText(canvas, row['pair_id'] + ' | raw portrait, no rotation', (8, 43), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 1)
            writer.write(canvas)
    finally:
        writer.release()
        for cap in caps:
            cap.release()
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', str(intermediate), '-c:v', 'libx264',
                    '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], check=True)
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(output)]))
    decoded = cv2.VideoCapture(str(output))
    decoded_frames = 0
    try:
        while True:
            ok, _ = decoded.read()
            if not ok:
                break
            decoded_frames += 1
    finally:
        decoded.release()
    actual_duration = float(probe['format']['duration'])
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    if decoded_frames != round(duration * fps) or abs(actual_duration - duration) > .001:
        raise RuntimeError(f"Invalid synchronization preview {output}: {decoded_frames} frames, {actual_duration}s")
    if video['codec_name'] != 'h264':
        raise RuntimeError(f"Unexpected preview codec: {video['codec_name']}")
    validation = dict(preview=str(output.relative_to(ROOT)), codec=video['codec_name'],
                      duration_s=actual_duration, decoded_frames=decoded_frames,
                      bytes=output.stat().st_size, width=video['width'], height=video['height'])
    return dict(path=str(output.relative_to(ROOT)), primary_start_s=start,
                secondary_start_s=start + row['offset_s'], duration_s=duration,
                frame_rate=fps, includes_audio=False, original_orientation_preserved=True,
                validation=validation)


def main():
    # One sign/normalization self-check catches a flipped offset implementation.
    rng = np.random.default_rng(123)
    toy = rng.normal(size=300)
    delayed = np.concatenate([rng.normal(size=17), toy])
    assert best_offsets(toy, delayed, range(-25, 26), 200)[0]['offset_s'] == .17
    manifest = {r['recording_id']: r for r in csv.DictReader((DATA / 'raw_manifest.csv').open())}
    intermediate = RESULTS / 'sync_intermediate'
    intermediate.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'synchronization_previews').mkdir(parents=True, exist_ok=True)
    envelopes = {}
    for rid, record in manifest.items():
        envelopes[rid] = audio_envelope(ROOT / record['relative_path'])
        np.save(intermediate / (rid + '_audio_log_rms_100hz.npy'), envelopes[rid])
    matrix = {}
    for p in range(5, 9):
        for s in range(1, 5):
            pe, se = envelopes[f'recording_{p:04d}'], envelopes[f'recording_{s:04d}']
            minimum = min(20 * RATE, int(min(len(pe), len(se)) * .75))
            matrix[f'primary_{p:04d}_secondary_{s:04d}'] = best_offsets(pe, se, range(-9000, 9001), minimum)
    rows, pair_evidence = [], []
    for number, (p, s, anchors) in enumerate(PAIRS, 1):
        pr, sr = f'recording_{p:04d}', f'recording_{s:04d}'
        peak = matrix[f'primary_{p:04d}_secondary_{s:04d}'][0]
        offset = peak['offset_s']
        row = dict(pair_id=f'pair_{number:04d}', primary_recording_id=pr, secondary_recording_id=sr,
                   primary_filename=manifest[pr]['filename'], secondary_filename=manifest[sr]['filename'],
                   offset_s=offset, uncertainty_s=.25 if anchors else .05,
                   synchronization_method='visual_multi_event_review+audio_log_rms_crosscorrelation' if anchors else 'audio_log_rms_crosscorrelation_only',
                   confidence='high' if anchors else 'high_audio_pairing_no_visual_motion',
                   audio_correlation=peak['correlation'], offset_convention='secondary_time = primary_time + offset_s',
                   unresolved_ambiguity='Phone A/V latency not independently calibrated; visual precision estimated at sampled-frame resolution.' if anchors else 'Stationary scene has no shared motion anchors. Audio pairing supported; visual alignment unverified.')
        pe, se = envelopes[pr], envelopes[sr]
        lo, hi = max(0., -offset), min(len(pe) / RATE, len(se) / RATE - offset)
        edges = np.linspace(lo + .5, hi - .5, 4)
        chunks = []
        for a, b in zip(edges[:-1], edges[1:]):
            local = best_offsets(pe, se, range(round(offset * RATE) - RATE, round(offset * RATE) + RATE + 1),
                                 int((b - a) * RATE * .8), round(a * RATE), round(b * RATE))[0]
            chunks.append(dict(primary_start_s=float(a), primary_end_s=float(b), **local))
        evidence = dict(**row, audio_thirds=chunks,
                        observed_offset_range_s=max(c['offset_s'] for c in chunks) - min(c['offset_s'] for c in chunks),
                        visual_corroboration=dict(reviewed_by='agent_inspector_1_3', human_review=False,
                                                 method='Matched grasp/lift/place/reset states viewed in paired source frames; mapped times use the audio offset and are visual corroboration, not independently fitted anchors.',
                                                 anchors=[dict(primary_s=round(t - offset, 3), secondary_s=t) for t in anchors]),
                        uncertainty_interpretation='Estimated visual alignment precision, not a measured A/V latency bound.' if anchors else 'Audio offset estimate only. Visual timing uncertainty is unknown.',
                        preview=preview(ROOT / manifest[pr]['relative_path'], ROOT / manifest[sr]['relative_path'], row))
        rows.append(row)
        pair_evidence.append(evidence)
        print(row['pair_id'], pr, sr, offset, peak['correlation'], flush=True)
    with (DATA / 'camera_pairs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = dict(offset_convention='secondary_time = primary_time + offset_s',
                  audio_method=dict(sample_rate_hz=8000, envelope_rate_hz=RATE,
                                    envelope='log(root mean square of mono amplitude + 1e-5)',
                                    search_range_s=[-90, 90], minimum_overlap='min(20 seconds, 75% of shorter recording)',
                                    peaks_separated_by_s=.5, used_audio_as='Support after visual candidate pairing; static pair uses audio only.'),
                  candidates_all_4_by_4=matrix, pairs=pair_evidence,
                  command='.venv-sim/bin/python scripts/m0_sync.py',
                  versions=dict(opencv=cv2.__version__, numpy=np.__version__),
                  human_review_complete=False)
    (RESULTS / 'synchronization_evidence.json').write_text(json.dumps(output, indent=2) + '\n')
    validations = [p['preview']['validation'] for p in pair_evidence]
    (RESULTS / 'synchronization_preview_validation.json').write_text(json.dumps(validations, indent=2) + '\n')


if __name__ == '__main__':
    main()
