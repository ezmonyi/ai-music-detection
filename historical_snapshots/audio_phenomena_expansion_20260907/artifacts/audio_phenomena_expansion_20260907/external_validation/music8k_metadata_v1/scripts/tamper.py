import os
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm


def load_audio(path: str):
    """
    读取音频。

    librosa.load(mono=False):
    - 单声道: shape = (n_samples,)
    - 多声道: shape = (channels, n_samples)
    """
    y, sr = librosa.load(path, sr=None, mono=False)
    return y, sr


def save_audio(path: str, y: np.ndarray, sr: int):
    """
    保存音频为 wav。

    soundfile 写入格式:
    - 单声道: shape = (n_samples,)
    - 多声道: shape = (n_samples, channels)

    librosa 读取多声道时是 (channels, n_samples)，
    因此保存前需要转置。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if y.ndim == 2:
        y = y.T

    sf.write(path, y, sr)


def make_output_path(input_path: str, output_dir: str, suffix: str, use_hash: bool = True) -> str:
    """
    根据原始文件名生成新的输出路径。

    例如:
    song.wav -> song_pitchshift.wav
    song.mp3 -> song_pitchshift.wav

    如果 use_hash=True:
    song.wav -> song_pitchshift_a1b2c3d4.wav

    加 hash 是为了避免不同目录下存在同名文件时互相覆盖。
    """
    p = Path(input_path)

    if use_hash:
        h = hashlib.md5(str(p).encode("utf-8")).hexdigest()[:8]
        new_name = f"{p.stem}_{suffix}_{h}.wav"
    else:
        new_name = f"{p.stem}_{suffix}.wav"

    return os.path.join(output_dir, new_name)


def pitch_shift_audio(y: np.ndarray, sr: int, min_steps: float = -2.0, max_steps: float = 2.0):
    """
    随机变调，范围为 [-2, +2] semitones。
    """
    n_steps = np.random.uniform(min_steps, max_steps)
    y_out = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    return y_out


def time_stretch_audio(y: np.ndarray, min_rate: float = 0.8, max_rate: float = 1.2):
    """
    随机 time-stretch，范围为 [0.8, 1.2]。

    rate < 1.0: 音频变慢、变长
    rate > 1.0: 音频变快、变短
    """
    rate = np.random.uniform(min_rate, max_rate)
    y_out = librosa.effects.time_stretch(y=y, rate=rate)
    return y_out


def add_white_noise(y: np.ndarray, snr_db: float = 20.0):
    """
    按指定 SNR 添加白噪声。

    SNR 越小，噪声越强。
    例如:
    - 20 dB: 较轻噪声
    - 10 dB: 更明显噪声
    """
    y = y.astype(np.float32)

    signal_power = np.mean(y ** 2)

    if signal_power <= 1e-12:
        return y

    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise = np.random.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=y.shape
    ).astype(np.float32)

    y_out = y + noise

    # 防止写出时爆音
    y_out = np.clip(y_out, -1.0, 1.0)

    return y_out.astype(np.float32)


def process_one_audio(
    input_path: str,
    output_dir: str,
    operation: str,
    noise_snr_db: float = 20.0,
    use_hash: bool = True,
):
    """
    对单个音频文件执行一种篡改操作，并返回新音频路径。
    """
    input_path = str(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    suffix_map = {
        "pitchshift": "pitchshift",
        "stretch": "stretch",
        "noise": "noise",
    }

    if operation not in suffix_map:
        raise ValueError(f"Unknown operation: {operation}")

    # If output_dir is empty or None, save next to source audio file
    if not output_dir:
        out_dir = str(Path(input_path).parent)
    else:
        out_dir = output_dir

    output_path = make_output_path(
        input_path=input_path,
        output_dir=out_dir,
        suffix=suffix_map[operation],
        use_hash=use_hash,
    )

    y, sr = load_audio(input_path)

    if operation == "pitchshift":
        y_out = pitch_shift_audio(y, sr)

    elif operation == "stretch":
        y_out = time_stretch_audio(y)

    elif operation == "noise":
        y_out = add_white_noise(y, snr_db=noise_snr_db)

    else:
        raise ValueError(f"Unknown operation: {operation}")

    save_audio(output_path, y_out, sr)

    return output_path


def process_csv(
    input_csv: str,
    operation: str,
    output_csv: str,
    output_audio_dir: str,
    noise_snr_db: float = 20.0,
    use_hash: bool = True,
    max_files: int = None,
):
    """
    读取 input.csv，对 full_path 和 vocal_path 两列音频执行指定操作，
    然后保存新的 csv 文件。

    label 和 source 保持不变。
    """
    df = pd.read_csv(input_csv)

    required_cols = ["full_path", "vocal_path", "label", "source"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df_new = df.copy()

    # 避免同一个音频路径重复处理
    cache = {}

    processed_count = 0
    done = False

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {operation}"):
        for col in ["full_path", "vocal_path"]:
            old_path = row[col]

            if pd.isna(old_path) or str(old_path).strip() == "":
                df_new.at[idx, col] = old_path
                continue

            old_path = str(old_path)

            cache_key = (old_path, operation)

            if cache_key in cache:
                new_path = cache[cache_key]
            else:
                if max_files is not None and processed_count >= max_files:
                    done = True
                    break

                new_path = process_one_audio(
                    input_path=old_path,
                    output_dir=output_audio_dir,
                    operation=operation,
                    noise_snr_db=noise_snr_db,
                    use_hash=use_hash,
                )
                cache[cache_key] = new_path
                processed_count += 1

            df_new.at[idx, col] = new_path

        if done:
            break

    df_new.to_csv(output_csv, index=False)

    print(f"Saved CSV: {output_csv}")
    if output_audio_dir:
        print(f"Saved audio dir: {output_audio_dir}")
    else:
        print("Saved audio files next to original source files")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        default="input.csv",
        help="Input CSV file with columns: full_path, vocal_path, label, source",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Root directory for saving tampered audio files. If empty, tampered audio are saved next to source audio files.",
    )

    parser.add_argument(
        "--output_csv_dir",
        type=str,
        default="",
        help="Directory for saving output CSV files. If empty, save next to input CSV.",
    )

    parser.add_argument(
        "--noise_snr_db",
        type=float,
        default=20.0,
        help="SNR in dB for white noise addition. Lower value means stronger noise.",
    )

    parser.add_argument(
        "--no_hash",
        action="store_true",
        help="Disable hash suffix in output filenames. Not recommended if filenames may duplicate.",
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of unique audio files to process per CSV (default: no limit)",
    )

    args = parser.parse_args()

    input_csv = args.input_csv
    output_root = args.output_root
    output_csv_dir = args.output_csv_dir
    use_hash = not args.no_hash
    max_files = args.max_files

    # 根据输入 CSV 名称生成输出 csv 文件名
    base = Path(input_csv).stem
    csv_parent = str(Path(input_csv).parent)

    if output_csv_dir:
        os.makedirs(output_csv_dir, exist_ok=True)
        csv_out_dir = output_csv_dir
    else:
        csv_out_dir = csv_parent

    # If output_root is provided, create per-operation subfolders there; otherwise
    # tampered audio files will be saved next to the source audio files.
    if output_root:
        os.makedirs(output_root, exist_ok=True)

    tasks = [
        {
            "operation": "pitchshift",
            "output_csv": os.path.join(csv_out_dir, f"{base}_pitchshift.csv"),
            "output_audio_dir": os.path.join(output_root, "audio_pitchshift") if output_root else "",
        },
        {
            "operation": "stretch",
            "output_csv": os.path.join(csv_out_dir, f"{base}_stretch.csv"),
            "output_audio_dir": os.path.join(output_root, "audio_stretch") if output_root else "",
        },
        {
            "operation": "noise",
            "output_csv": os.path.join(csv_out_dir, f"{base}_noise.csv"),
            "output_audio_dir": os.path.join(output_root, "audio_noise") if output_root else "",
        },
    ]

    for task in tasks:
        # create output_audio_dir only if specified
        if task["output_audio_dir"]:
            os.makedirs(task["output_audio_dir"], exist_ok=True)

        process_csv(
            input_csv=input_csv,
            operation=task["operation"],
            output_csv=task["output_csv"],
            output_audio_dir=task["output_audio_dir"],
            noise_snr_db=args.noise_snr_db,
            use_hash=use_hash,
            max_files=max_files,
        )


if __name__ == "__main__":
    main()