#!/usr/bin/env python3
"""Static scientific PNGs from frozen paired controls; no detection fits."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'results/stereo_music_controls_v1'
REPORT = ROOT/'results/stereo_music_controls_report_v1'
OUT = ROOT/'results/stereo_music_controls_figures_v1'
CODECS = ['mp3_128k', 'mp3_256k', 'm4a_128k', 'm4a_256k']
BANDS = [(80,500),(500,2000),(2000,6000),(6000,12000),(12000,20000)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    for root in [RAW, REPORT]:
        commit = json.loads((root/'COMMIT.json').read_text())
        for rel, expected in commit['products'].items():
            p = root/rel
            assert p.stat().st_size == expected['bytes'] and sha(p)==expected['sha256']
    data = json.loads((REPORT/'report_data.json').read_text())
    OUT.mkdir(exist_ok=False)
    labels = [f'{lo/1000:g}-{hi/1000:g}' + ('*' if hi==20000 else '') for lo,hi in BANDS]
    table = [r for r in data['codec_table'] if r['metric']=='side_energy_fraction']
    delta = np.array([[next(r['codec_delta_median'] for r in table
        if r['codec']==c and r['band_low']==lo) for lo,hi in BANDS] for c in CODECS])
    ratio = np.array([[next(r['ratio_median'] for r in table
        if r['codec']==c and r['band_low']==lo) for lo,hi in BANDS] for c in CODECS])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, matrix, title, hi in zip(axes, [delta, ratio],
        ['Absolute side-fraction change', 'Codec / width x2 response'], [.3, 1.]):
        im = ax.imshow(matrix, vmin=0, vmax=hi, cmap='magma', aspect='auto')
        ax.set_xticks(range(5), labels, fontsize=10)
        ax.set_yticks(range(4), ['MP3 128k','MP3 256k','AAC 128k','AAC 256k'])
        ax.set_xlabel('Frequency band (kHz); * exploratory')
        ax.set_title(title)
        for i in range(4):
            for j in range(5):
                ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',
                    color='black' if matrix[i,j]>.65*hi else 'white')
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle('Stereo width descriptors can reflect codec processing\n'
        'Median across 24 paired recording controls; not AI/Human accuracy', fontsize=14)
    fig.savefig(OUT/'codec_width_sensitivity.png', dpi=180)
    plt.close(fig)

    stacks = [[], []]
    for p in sorted(RAW.glob('*/comparisons.json')):
        directory = p.parent
        with np.load(directory/'original_frames.npz') as z:
            original = z['side_energy_fraction']
        with np.load(directory/'width_2_frames.npz') as z:
            width = z['side_energy_fraction']
        with np.load(directory/'m4a_128k_reference_frames.npz') as z:
            reference = z['side_energy_fraction']
        with np.load(directory/'m4a_128k_frames.npz') as z:
            codec = z['side_energy_fraction']
        assert original.shape==width.shape==reference.shape==codec.shape
        stacks[0].append(width-original)
        stacks[1].append(codec-reference)
    means, counts = [], []
    for stack in stacks:
        x = np.stack(stack)
        n = np.isfinite(x).sum(axis=0)
        means.append(np.divide(np.nansum(x,axis=0), n,
            out=np.full(n.shape,np.nan,dtype=float), where=n>0))
        counts.append(n)
    np.savez_compressed(OUT/'paired_frame_mean_data.npz', width_difference=means[0],
        aac128_difference=means[1], width_valid_recordings=counts[0], aac_valid_recordings=counts[1])
    hop = 1024/44100
    first = 2048/44100
    extent = [first-hop/2, first+(means[0].shape[1]-.5)*hop, -.5, 4.5]
    fig, axes = plt.subplots(2,1,figsize=(12,7),layout='constrained',sharex=True)
    for ax, values, title in zip(axes, means,
        ['Known M/S width x2 minus original', 'AAC128 round trip minus aligned original']):
        im = ax.imshow(values, origin='lower', aspect='auto', extent=extent,
            cmap='RdBu_r', vmin=-.35, vmax=.35, interpolation='nearest')
        ax.set_yticks(range(5), labels)
        ax.set_ylabel('Band (kHz)')
        ax.set_title(title)
        fig.colorbar(im,ax=ax,label='Mean paired side-fraction difference',shrink=.9)
    axes[-1].set_xlabel('STFT frame-center time within excerpt (seconds)')
    fig.suptitle('Band-by-frame spatial difference heatmaps (24 recordings)\n'
        'Relative excerpt time is not beat/phrase aligned; *12-20 kHz is exploratory',fontsize=13)
    fig.savefig(OUT/'paired_band_frame_heatmaps.png',dpi=180)
    plt.close(fig)
    receipt = dict(raw_commit_sha256=sha(RAW/'COMMIT.json'),
        report_commit_sha256=sha(REPORT/'COMMIT.json'), plotter_sha256=sha(__file__),
        records=len(stacks[0]), frames=means[0].shape[1],
        minimum_recordings_per_cell=[int(n.min()) for n in counts],
        maximum_recordings_per_cell=[int(n.max()) for n in counts],
        heatmap_scope='pointwise mean of paired changes at relative excerpt time; not a removal template',
        products={p.name:dict(sha256=sha(p),bytes=p.stat().st_size)
                  for p in sorted(OUT.iterdir()) if p.is_file()})
    with (OUT/'COMMIT.json').open('x') as f:
        json.dump(receipt,f,indent=2,sort_keys=True)
        f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
