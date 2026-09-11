#!/usr/bin/env python3
"""Frozen external-only leave-singer-out breath baseline; never AI labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.fft import dct
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly
import librosa
import soundfile as sf
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

SEED = 20260907
HOP, SR, NFFT = 160, 16000, 512
FEATURE_NAMES = ([f"mfcc_{i}" for i in range(1, 13)] +
                 ["relative_rms_db", "flatness", "centroid_fraction", "bandwidth_fraction",
                  "band_80_1000", "band_1000_3000", "band_3000_7500", "zcr", "periodicity"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def targets(times, label):
    truth = np.zeros(len(times), dtype=np.int8)
    valid = np.ones(len(times), dtype=bool)
    events = []
    for event in label.get("breath_events", []):
        a, b = event["start_sec"], event["end_sec"]
        if event.get("confidence") in ("high", "medium"):
            events.append((a, b))
        else:
            valid[(times >= a - .02) & (times <= b + .02)] = False
    merged = []
    for a, b in sorted(events):
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a,b))
    for a,b in merged:
        truth[(times >= a) & (times < b)] = 1
        valid[(np.abs(times - a) <= .02) | (np.abs(times - b) <= .02)] = False
    for kind in ("silent_breaths", "uncertain", "exhales"):
        for event in label.get(kind, []):
            valid[(times >= event["start_sec"] - .02) & (times <= event["end_sec"] + .02)] = False
    return truth, valid, merged


def predicted_events(scores, valid, times):
    active = (scores >= .5) & valid
    # Fill only bounded gaps <=3 frames, never ignored intervals or edge padding.
    indices = np.flatnonzero(active)
    for left, right in zip(indices[:-1], indices[1:]):
        if 1 < right - left <= 4 and valid[left:right + 1].all():
            active[left:right + 1] = True
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    result = []
    for a, b in zip(starts, stops):
        duration = (b - a) * .01
        # Sustained false alarms must not vanish. Overlong runs are retained
        # and cannot match a reference; they count as false positive events.
        if .05 - 1e-8 <= duration:
            result.append((max(0., float(times[a]) - .005), float(times[b-1]) + .005))
    return result


def match_events(reference, prediction):
    if not reference or not prediction:
        return 0, len(prediction), len(reference)
    valid = np.array([[d-c <= 2 + 1e-8 and abs(a-c) <= .2 + 1e-9 and min(b,d) > max(a,c)
                       for c,d in prediction] for a,b in reference], dtype=np.int8)
    ri, pi = linear_sum_assignment(-valid)
    tp = int(valid[ri, pi].sum())
    return tp, len(prediction) - tp, len(reference) - tp


def event_metrics(tp, fp, fn):
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def extract(job):
    row, labels_dir, audio_dir = job
    name = row["filename"]
    audio_path = audio_dir / name
    label_path = labels_dir / "labels/vocalset" / (Path(name).stem + ".breath.json")
    if sha(audio_path) != row["audio_sha256"] or sha(label_path) != row["annotation_sha256"]:
        raise ValueError("Audio/annotation checksum mismatch: " + name)
    label = json.loads(label_path.read_text())
    if not math.isfinite(label["duration_sec"]) or label["duration_sec"] <= 0:
        raise ValueError("Invalid annotation duration")
    y, sr = sf.read(audio_path, always_2d=True, dtype="float64")
    y = y.mean(axis=1)
    y -= y.mean()
    if sr != SR:
        divisor = math.gcd(sr, SR)
        y = resample_poly(y, SR//divisor, sr//divisor, window=("kaiser", 5.0))
    y /= max(np.sqrt(np.mean(y*y)), 1e-12)
    framed = np.lib.stride_tricks.sliding_window_view(np.pad(y, (256, 256)), NFFT)[::HOP]
    window = .5 - .5*np.cos(2*np.pi*np.arange(NFFT)/NFFT)
    spectrum = np.fft.rfft(framed * window, axis=1)
    power = np.abs(spectrum)**2
    frequency = np.fft.rfftfreq(NFFT, 1/SR)
    normalized = (power + 1e-12)/(power.sum(axis=1, keepdims=True) + 1e-12*power.shape[1])
    centroid = (normalized*frequency).sum(axis=1)
    mel = power @ librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=24, fmin=80, fmax=7500).T
    logmel = 10*np.log10(np.maximum(mel, 1e-12))
    logmel = np.maximum(logmel, np.max(logmel)-80)
    cepstrum = dct(logmel, type=2, norm="ortho", axis=1)[:, 1:13]
    autocorr = np.fft.irfft(power, n=NFFT, axis=1)
    periodicity = autocorr[:,16:251].max(axis=1)/np.maximum(autocorr[:,0],1e-12)
    extras = [10*np.log10(np.maximum((framed*framed).mean(axis=1),1e-12)),
              np.exp(np.log(power+1e-12).mean(axis=1))/np.maximum(power.mean(axis=1),1e-12),
              centroid/8000, np.sqrt((normalized*(frequency[None,:]-centroid[:,None])**2).sum(axis=1))/8000]
    extras += [normalized[:,(frequency>=a)&(frequency<b)].sum(axis=1) for a,b in ((80,1000),(1000,3000),(3000,7500))]
    extras += [np.mean(np.diff(framed>=0,axis=1),axis=1),periodicity]
    features = np.column_stack([cepstrum,*extras]).astype(np.float32)
    times = np.arange(len(features))*.01
    truth, valid, events = targets(times,label)
    valid &= times < len(y)/SR
    hard_negative = np.zeros(len(times),dtype=bool)
    for event in label.get("hard_negatives",[]):
        hard_negative |= (times>=event["start_sec"]) & (times<event["end_sec"])
    hard_negative &= valid & (truth==0)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite frame measurements: " + name)
    return {"name":name,"singer":row["singer"],"features":features,"times":times,
            "truth":truth,"valid":valid,"events":events,"duration":len(y)/SR,
            "hard_negative":hard_negative}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Refusing existing output; preserve all receipts")
    acquisition = json.loads((args.acquisition_dir/"acquisition_summary.json").read_text())
    if acquisition["errors"] or acquisition["downloaded_verified"] != acquisition["selected"]:
        raise ValueError("Primary benchmark requires complete acquisition accounting without failed files")
    if len({r["filename"] for r in acquisition["manifest"]}) != len(acquisition["manifest"]):
        raise ValueError("Duplicate acquisition identities")
    selected, excluded = [], []
    for row in acquisition["manifest"]:
        label_path = args.labels_dir/"labels/vocalset"/(Path(row["filename"]).stem+".breath.json")
        label = json.loads(label_path.read_text())
        if label.get("labeler","").strip() and label.get("review_time_sec",0)>0:
            selected.append(row)
        else:
            excluded.append({"filename":row["filename"],"reason":"No named reviewer or review time"})
    if len(set(r["singer"] for r in selected)) < 5:
        raise ValueError("Insufficient primary singer coverage")
    args.output_dir.mkdir(parents=True)
    config = {"seed":SEED,"code_sha256":sha(__file__),"protocol_sha256":sha(args.protocol),
              "acquisition_sha256":sha(args.acquisition_dir/"acquisition_summary.json"),
              "selected_primary_files":[r["filename"] for r in selected],"excluded":excluded,
              "frame_length":NFFT,"hop_length":HOP,"sample_rate":SR,"features":FEATURE_NAMES,
              "classifier":{"name":"LogisticRegression","C":1,"class_weight":"balanced","max_iter":1000,"threshold":.5},
              "gate":{"event_f1":.70,"precision":.70},"no_ai_human_labels":True,
              "runtime":{"numpy":np.__version__,"librosa":librosa.__version__,
                         "scipy":scipy.__version__,"sklearn":sklearn.__version__,"soundfile":sf.__version__}}
    write_json(args.output_dir/"frozen_config.json",config)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        clips = list(pool.map(extract, [(r,args.labels_dir,args.acquisition_dir/"audio") for r in selected]))
    (args.output_dir/"frames").mkdir()
    for clip in clips:
        np.savez_compressed(args.output_dir/"frames"/(clip["name"]+".npz"), **{k:clip[k] for k in ("features","times","truth","valid","hard_negative")})
    records, models = [], []
    for singer in sorted(set(c["singer"] for c in clips)):
        train_x, train_y = [], []
        for clip in clips:
            if clip["singer"] == singer:
                continue
            positive = np.flatnonzero(clip["valid"] & (clip["truth"]==1))
            negative = np.flatnonzero(clip["valid"] & (clip["truth"]==0))
            if len(negative)>2000:
                rng = np.random.default_rng(SEED+int(hashlib.sha256(clip["name"].encode()).hexdigest()[:8],16))
                negative = rng.choice(negative,2000,replace=False)
            use = np.r_[positive,negative]
            train_x.append(clip["features"][use]); train_y.append(clip["truth"][use])
        x, y = np.concatenate(train_x),np.concatenate(train_y)
        if len(np.unique(y)) != 2:
            raise ValueError("Training fold lacks both external event classes")
        scale = StandardScaler().fit(x)
        model = LogisticRegression(C=1,class_weight="balanced",max_iter=1000,random_state=SEED).fit(scale.transform(x),y)
        models.append({"heldout_singer":singer,"mean":scale.mean_.tolist(),"scale":scale.scale_.tolist(),
                       "coef":model.coef_.tolist(),"intercept":model.intercept_.tolist(),"iterations":model.n_iter_.tolist(),
                       "training_frames":len(y),"positive_training_frames":int(y.sum())})
        for clip in clips:
            if clip["singer"] != singer:
                continue
            score = model.predict_proba(scale.transform(clip["features"]))[:,1]
            predicted = predicted_events(score,clip["valid"],clip["times"])
            tp,fp,fn = match_events(clip["events"],predicted)
            valid = clip["valid"]
            ba = balanced_accuracy_score(clip["truth"][valid],score[valid]>=.5) if len(np.unique(clip["truth"][valid]))==2 else None
            record = {"filename":clip["name"],"singer":singer,**event_metrics(tp,fp,fn),"frame_ba":ba,
                      "duration":clip["duration"],"ignored_seconds":float((~valid).sum()*.01),
                      "valid_frames":int(valid.sum()),"positive_valid_frames":int(clip["truth"][valid].sum()),
                      "tp_frames":int(((score>=.5)&valid&(clip["truth"]==1)).sum()),
                      "tn_frames":int(((score<.5)&valid&(clip["truth"]==0)).sum()),
                      "hard_negative_frames":int(clip["hard_negative"].sum()),
                      "false_positive_hard_negative_frames":int(((score>=.5)&clip["hard_negative"]).sum()),
                      "reference_events":clip["events"],"predicted_events":predicted}
            records.append(record)
            np.savez_compressed(args.output_dir/"frames"/(clip["name"]+".predictions.npz"),score=score)
        print(json.dumps({"singer_finished":singer,"clips_finished":len(records)}),flush=True)
    totals = event_metrics(*[sum(r[k] for r in records) for k in ("tp","fp","fn")])
    positives = sum(r["positive_valid_frames"] for r in records)
    negatives = sum(r["valid_frames"]-r["positive_valid_frames"] for r in records)
    frame_ba = .5*(sum(r["tp_frames"] for r in records)/positives + sum(r["tn_frames"] for r in records)/negatives)
    summary = {"status":"completed","primary_clips":len(records),"singers":len(models),"overall":totals,
               "strict_positive_events_after_overlap_merge":sum(len(r["reference_events"]) for r in records),
               "no_event_primary_clips":sum(not r["reference_events"] for r in records),
               "ignored_primary_seconds":sum(r["ignored_seconds"] for r in records),
               "primary_duration_seconds":sum(r["duration"] for r in records),
               "aggregate_frame_balanced_accuracy":frame_ba,
               "excluded_acquisition_clip_count":len(excluded),
               "hard_negative_frames":sum(r["hard_negative_frames"] for r in records),
               "false_positive_hard_negative_frames":sum(r["false_positive_hard_negative_frames"] for r in records),
               "gate_passed":len(records)==len(selected) and (totals["f1"] or 0)>=.7 and (totals["precision"] or 0)>=.7,
               "validates_ai_human_classification":False,"excluded":excluded,"acquisition_errors":acquisition["errors"],
               "per_singer":{s:event_metrics(*[sum(r[k] for r in records if r["singer"]==s) for k in ("tp","fp","fn")]) for s in sorted(set(c["singer"] for c in clips))}}
    write_json(args.output_dir/"event_predictions.json",records)
    write_json(args.output_dir/"fold_models.json",models)
    write_json(args.output_dir/"summary.json",summary)
    write_json(args.output_dir/"artifact_sha256.json",{str(p.relative_to(args.output_dir)):sha(p) for p in sorted(args.output_dir.rglob("*")) if p.is_file()})
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
