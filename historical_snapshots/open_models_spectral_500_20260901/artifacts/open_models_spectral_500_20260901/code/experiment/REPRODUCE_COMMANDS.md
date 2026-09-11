# HeartMuLa / ACE-Step 500×2 复现命令

以下命令对应 2026-09-01 冻结实验；省略 SSH 登录本身，不包含任何密钥。所有模型推理和 Demucs 均在 `5090-4` 的 RTX 5090 上完成。

```bash
exp=/mnt/nfs-code/users/yi/open_models_spectral_500_20260901
baseline=/mnt/nfs-code/users/yi/demucs_bias_corrected_1000_20260901
```

## 冻结 prompt 清单

Muse metadata 通过 Hugging Face CLI 以 revision `b1bf3bf906daab3a896e14f6dea58cc295848452` 下载，仅保留 `train_cn.jsonl`、`train_en.jsonl` 文本；未下载音频。

```bash
$exp/code/ACE-Step-1.5/.venv/bin/python \
  $exp/code/experiment/build_muse_manifest.py \
  --train-cn $exp/prompts/train_cn.jsonl \
  --train-en $exp/prompts/train_en.jsonl \
  --output-dir $exp/prompts/frozen_500 \
  --seed 20260901
sha256sum $exp/prompts/frozen_500/prompt_manifest.jsonl
```

冻结 manifest SHA-256：`bb0941b91d41d10a978cb9e4b077399e89fea92526ee469e0e6743b09efb917b`。

## ACE-Step 1.5 推理

```bash
CUDA_VISIBLE_DEVICES=2 $exp/code/ACE-Step-1.5/.venv/bin/python \
  $exp/code/experiment/run_acestep_batch.py \
  --manifest $exp/prompts/frozen_500/prompt_manifest.jsonl \
  --checkpoint-dir $exp/models/acestep/Ace-Step1.5 \
  --output-dir $exp/generation/acestep/raw \
  --rendered-input-dir $exp/generation/acestep/rendered_inputs \
  --state-file $exp/state/acestep_generation.jsonl \
  --duration-seconds 30 --device cuda
```

脚本固定 Turbo 8 steps、thinking/CoT off、normalization off、DCW double/0.05/0.02/Haar，并逐条记录 seed、耗时、音频 metadata 与 SHA-256。

## HeartMuLa 推理

```bash
CUDA_VISIBLE_DEVICES=0,1 $exp/envs/heartmula/bin/python \
  $exp/code/experiment/run_heartmula_batch.py \
  --manifest $exp/prompts/frozen_500/prompt_manifest.jsonl \
  --model-root $exp/models/heartmula \
  --output-dir $exp/generation/heartmula/raw \
  --rendered-input-dir $exp/generation/heartmula/rendered_inputs \
  --state-file $exp/state/heartmula_generation_r0_49_134.jsonl \
  --start-index 49 --limit 86
```

其余重平衡 worker 同一命令，分别为：

- `CUDA_VISIBLE_DEVICES=2,3`，`--start-index 135 --limit 143`
- `CUDA_VISIBLE_DEVICES=4,5`，`--start-index 278 --limit 137`
- `CUDA_VISIBLE_DEVICES=6,7`，`--start-index 415 --limit 85`

这些区间覆盖当时缺失清单；区间内已有合法输出会校验后跳过。推理参数为 30,000 ms、top-k 50、temperature 1.0、CFG 1.5、MuLa bf16、codec fp32。

## 音频标准化与验证

若受控中断恰好发生在 atomic rename 之后、state append 之前，可先运行 `reconcile_generated_state.py`。它不改音频，只对 manifest 中“已有合法 target 但缺少 ok event”的条目重新计算 metadata/SHA-256，写入独立 `heartmula_generation_reconciled.jsonl`；历史 state 不覆盖。

```bash
$exp/code/ACE-Step-1.5/.venv/bin/python \
  $exp/code/experiment/standardize_generated_audio.py \
  --manifest $exp/prompts/frozen_500/prompt_manifest.jsonl \
  --generator heartmula \
  --input-dir $exp/generation/heartmula/raw \
  --output-dir $exp/standardized/heartmula \
  --validation-jsonl $exp/state/heartmula_standardization.jsonl
```

ACE-Step 只需把 `heartmula` 替换为 `acestep`。标准化固定 SoundFile + `scipy.signal.resample_poly`，输出双声道、44.1 kHz、30 秒、PCM-16 FLAC，不做响度归一化或 EQ。

```bash
$exp/code/ACE-Step-1.5/.venv/bin/python \
  $exp/code/experiment/validate_generated_batch.py \
  --manifest $exp/prompts/frozen_500/prompt_manifest.jsonl \
  --state-file $exp/state/heartmula_generation*.jsonl \
  --audio-dir $exp/generation/heartmula/raw \
  --generator heartmula --expected-sample-rate 48000 \
  --minimum-duration 30 --maximum-duration 30.2
```

上面的原始时长严格检查会因 42 条 HeartMuLa 提前结束而失败，失败 JSON 保留。随后用同一命令改为 `--minimum-duration 5 --maximum-duration 30.2` 验证模型原生输出完整性；统一长度由下一步 standardization 的尾部补零保证。

## Demucs vocal-only

```bash
$exp/code/ACE-Step-1.5/.venv/bin/python \
  $exp/code/experiment/run_demucs_generated.py \
  --input-dir $exp/standardized/heartmula \
  --output-dir $exp/demucs/heartmula \
  --python /mnt/nfs-code/users/yi/soulx_svc_cnceleb_moon10_260831/.venv/bin/python \
  --torch-home $exp/models/demucs \
  --state-file $exp/state/demucs_heartmula.jsonl \
  --batch-size 20 --gpu-index 3
```

脚本内部固定 `htdemucs --two-stems vocals --float32 --shifts 0 --overlap 0.25 --segment 7 --jobs 1`，并验证每个 stem 为 44.1 kHz、双声道、1,323,000 frames、float32 WAV。

## Human-vs-generator 与热图

```bash
$exp/code/ACE-Step-1.5/.venv/bin/python \
  $exp/code/experiment/prepare_analysis_layout.py \
  --prompt-manifest $exp/prompts/frozen_500/prompt_manifest.jsonl \
  --baseline-manifest $baseline/manifest.jsonl \
  --baseline-clips $baseline/input \
  --baseline-stems $baseline/output \
  --standardized-root $exp/standardized \
  --generated-stems-root $exp/demucs \
  --output-dir $exp/analysis/layouts \
  --generators heartmula

layout=$exp/analysis/layouts/human_vs_heartmula
PYTHONPATH=$exp/code/analysis_lib \
  $exp/code/ACE-Step-1.5/.venv/bin/python \
  $exp/code/analysis_lib/analyze_bias_corrected_vocals.py \
  --manifest $layout/manifest.jsonl --clips-dir $layout/clips \
  --stems-dir $layout/stems --bias-npz $exp/analysis/demucs_frequency_bias.npz \
  --output-dir $exp/analysis/human_vs_heartmula --seed 20260901 --workers 4

PYTHONPATH=$exp/code/analysis_lib \
  $exp/code/ACE-Step-1.5/.venv/bin/python \
  $exp/code/experiment/analyze_average_class_spectrogram.py \
  --manifest $layout/manifest.jsonl --stems-dir $layout/stems \
  --bias-npz $exp/analysis/demucs_frequency_bias.npz \
  --comparison-name "Human vs HeartMuLa" \
  --output-dir $exp/analysis/human_vs_heartmula
```

ACE-Step 使用同样命令与 `human_vs_acestep` layout。average map 的四个 panel 为 raw/corrected × absolute/frame-normalized shape；STFT 为 44.1 kHz、4096 FFT、1024 hop、300 Hz–20 kHz 聚合成 128 个对数频带。

## 配对与跨生成器分析

```bash
PYTHONPATH=$exp/code/analysis_lib \
  $exp/code/ACE-Step-1.5/.venv/bin/python \
  $exp/code/experiment/analyze_paired_generators.py \
  --prompt-manifest $exp/prompts/frozen_500/prompt_manifest.jsonl \
  --stems-root $exp/demucs --bias-npz $exp/analysis/demucs_frequency_bias.npz \
  --output-dir $exp/analysis/paired_generators
```

三模型汇总使用 `compare_generators.py`；单生成器 train/test matrix、leave-one-generator-out 与 pooled evaluation 使用 `cross_generator_generalization.py`。后者分别运行全量 cohort 和附加 `--active-only` 的 raw-frozen vocal-active sensitivity subset。二者均读取已冻结的 Suno baseline、HeartMuLa、ACE-Step manifest/metrics，不重新拟合 Demucs bias。
