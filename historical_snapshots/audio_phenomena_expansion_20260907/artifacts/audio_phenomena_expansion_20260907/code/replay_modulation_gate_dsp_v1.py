#!/usr/bin/env python3
"""Recompute every held-out Q view from audio, using the same numerical libraries.

Imports no producer/candidate. This is numerical implementation consistency,
not independent-library scientific or AI/human validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import butter, firwin, hilbert, resample_poly, sosfiltfilt


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(root, freeze_sha, output):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("new receipt outside immutable result required")
    commit_path = root / "COMMIT.json"
    commit_sha = sha(commit_path)
    commit = json.loads(commit_path.read_text())
    if (commit.get("status") != "committed" or commit.get("kind") != "heldout_measurement_gate_result"
            or commit.get("freeze_sha256") != freeze_sha or sha(root/"freeze.json") != freeze_sha):
        raise ValueError("committed gate/freeze linkage mismatch")
    products = commit["products"]
    if {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} != set(products)|{"COMMIT.json"}:
        raise ValueError("exact inventory mismatch")
    paths = sorted((root/"candidate_json").glob("*.json"))
    if len(paths) != 1248:
        raise ValueError("expected all 1248 view/condition/note measurements")
    fir = firwin(1281, 250, fs=16000, window=("kaiser",8.6))
    hann = .5-.5*np.cos(2*np.pi*np.arange(1000)/1000)
    frequency = np.arange(501,dtype=float)/2
    maximum = {"power":0., "input_energy":0., "band_energy":0., "envelope_mean":0.}
    views, comparisons = {}, 0
    def check(actual, saved, kind):
        np.testing.assert_allclose(actual,saved,rtol=1e-11,atol=1e-13)
        maximum[kind] = max(maximum[kind],float(np.max(np.abs(np.asarray(actual)-np.asarray(saved)))))
    for path in paths:
        relative = str(path.relative_to(root))
        if sha(path) != products[relative]["sha256"]:
            raise ValueError("measurement bytes changed")
        record = json.loads(path.read_text())
        wave_relative = record["analysis_waveform"]["path"]
        wave = root/wave_relative
        if (wave.is_symlink() or not wave.resolve().is_relative_to(root)
                or sha(wave) != record["analysis_waveform"]["sha256"]
                or sha(wave) != products[wave_relative]["sha256"]):
            raise ValueError("waveform linkage or bytes changed")
        sr,x = wavfile.read(wave)
        if sr!=16000 or x.shape!=(64000,) or x.dtype!=np.float64 or not np.all(np.isfinite(x)):
            raise ValueError("analysis waveform contract")
        result = record["measurement"]
        np.testing.assert_array_equal(frequency,result["frequency_hz"])
        if result["window_count"]!=1 or len(result["bands"])!=3:
            raise ValueError("measurement dimensions")
        peak = float(np.max(np.abs(x)))
        normalized = x/peak if peak else np.zeros(64000)
        for band_index,limits in enumerate(((250,1000),(1000,3000),(3000,7000))):
            filtered = sosfiltfilt(butter(4,limits,btype="bandpass",fs=16000,output="sos"),
                                   normalized,padtype="odd",padlen=27) if peak else np.zeros(64000)
            envelope = resample_poly(np.abs(hilbert(filtered)),1,32,window=fir,padtype="constant")[125:1125]
            mean = float(envelope.mean())
            if mean>0:
                fluctuation=envelope/mean-1
                fluctuation-=fluctuation.mean()
                power=np.abs(np.fft.rfft(fluctuation*hann))**2/(1000*np.sum(hann**2))
                power[1:-1]*=2
            else:
                power=np.zeros(501)
            window=result["bands"][band_index]["windows"][0]
            if window["start_seconds"]!=.25 or window["end_seconds"]!=2.25:
                raise ValueError("window position")
            check(power,window["modulation_power"],"power")
            check(float(np.mean(normalized[4000:36000]**2))*peak**2,window["input_mean_square"],"input_energy")
            check(float(np.mean(filtered[4000:36000]**2))*peak**2,window["band_mean_square"],"band_energy")
            check(mean*peak,window["envelope_mean"],"envelope_mean")
            comparisons+=1
        views[record["view"]]=views.get(record["view"],0)+1
    if views!={"original":416,"mp3_128k":416,"aac_lc_64k":416} or sha(commit_path)!=commit_sha:
        raise ValueError("view coverage/publication changed")
    receipt={"passed":True,"scope":"all_heldout_gate_waveform_DSP_replay_same_numerical_libraries",
             "code_sha256":sha(__file__),"freeze_sha256":freeze_sha,"result_commit_sha256":commit_sha,
             "views_replayed":views,"band_windows_replayed":comparisons,"spectrum_bins_replayed":comparisons*501,
             "maximum_absolute_errors":maximum,"rtol":1e-11,"atol":1e-13,
             "producer_imported":False,"criterion_decisions_recomputed":False,
             "external_gate_passed":False,"classifier_admitted":False,
             "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__}
    with output.open("x") as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False); stream.write("\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir",required=True)
    parser.add_argument("--freeze-sha256",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    audit(args.result_dir,args.freeze_sha256,args.output)
