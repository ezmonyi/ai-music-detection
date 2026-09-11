# Suno / HeartMuLa / ACE-Step 人声频域实验阶段报告

日期：2026-09-01

## 一、实验回答了什么

本轮把既有 Human 500 / Suno 500 扩展为三个 AI 生成器域：Suno、HeartMuLa、ACE-Step 各 500。HeartMuLa 与 ACE-Step 逐条共享同一组 Muse style/lyrics、语言、split 和整数 seed；全部分析输入统一为 30 秒、44.1 kHz、双声道、PCM-16 FLAC，再用同一 htdemucs vocal-only 参数分轨。

确认性分类只在 development 400+400 上拟合，在 locked test 100+100 上评估。Demucs correction 是实验开始前冻结的外部 MUSDB-oracle average frequency response，不用本轮 AI/Human 标签重新估计。

## 二、最重要的总体结论

“AI 人声的高频不自然”可以成为有效 heuristic，但它不是一个固定方向、固定频带的普适规则。三个生成器都在 10–16 kHz 显著偏离 Human，但 16–20 kHz 的方向完全不同：

| Generator | 0.3–5 kHz | 5–10 kHz | 10–16 kHz | 16–20 kHz |
|---|---:|---:|---:|---:|
| Suno | −1.158 dB | +1.812 dB | +4.322 dB | +0.146 dB |
| HeartMuLa | −1.469 dB | +1.917 dB | +14.104 dB | −15.912 dB |
| ACE-Step | −2.866 dB | +2.588 dB | +8.465 dB | +10.519 dB |

数值是 locked-test、bias-corrected、frame-normalized vocal spectral shape 的 AI−Human 平均值。HeartMuLa 在约 16.5 kHz 出现很陡的带宽边缘，随后一直低到约 −20 dB；ACE-Step 则在 16–20 kHz 持续过量。两者都可能让人听出“高频质感异常”，但前者更像 codec/bandwidth cutoff，后者更像超高频 excess。

因此 applied filter 更适合做成“两层”：先用跨模型较稳定的宽带/5–16 kHz heuristic，再保留 generator-specific 的 cutoff、air band、纹理和时序 fingerprint；不能只用“高频越多越 AI”。

## 三、单生成器 locked-test 结果

| Generator | 最佳 corrected frequency representation | ROC-AUC | 95% CI | Balanced accuracy |
|---|---|---:|---:|---:|
| Suno | without 5–10 kHz | 0.922 | 0.882–0.957 | 0.855 |
| HeartMuLa | full 0.3–20 kHz | 1.000 | 1.000–1.000 | 1.000 |
| ACE-Step | without 5–10 kHz | 0.988 | 0.971–0.999 | 0.970 |

窄带结果同样很强：HeartMuLa 5–10 kHz 0.9999、5–16 kHz 0.9998；ACE-Step 5–10 kHz 0.9713、5–16 kHz 0.9841；Suno 分别 0.8776、0.8992。可是三者最佳结果都不是“only 5–10 kHz”，说明高频听感有信息，但宽带谱形、中低频相对能量和时间结构仍有增益。

最稳定且可解释的 scalar cue 家族包括：

- `hf_tilt_5_16k_db_oct`、`hf_ratio_5_16_db`、`air_ratio_12_20_db`：高频亮度、倾斜与空气感/带宽边缘。
- `hf_crest_db`、`hf_mod_4_12_share`、`sibilance_burst_rate_hz`：高频纹理、调制和齿音 burst。
- `hnr_proxy_db`、`pitch_confidence`、`vibrato_depth_cents`：生成歌声过度周期性、音高稳定性和 vibrato 结构。

## 四、为什么单域 AUC 不能直接当作检测器性能

单生成器训练到另一个生成器时，AUC 有明显不对称，甚至低于 0.5，说明模型会学到方向相反的 generator fingerprint。例如 corrected `without 5–10 kHz`：Suno→HeartMuLa 0.905，但 HeartMuLa→Suno 0.379；HeartMuLa→ACE-Step 0.326，ACE-Step→HeartMuLa 0.593。

更严格的 leave-one-generator-out 结果为：

| Target generator | All tracks | Vocal-active subset |
|---|---:|---:|
| Suno | 0.698 | 0.702 |
| HeartMuLa | 0.949 | 0.942 |
| ACE-Step | 0.834 | 0.935 |

训练包含全部三个生成器时，pooled locked AUC 为 0.943；vocal-active pooled AUC 为 0.969。结论是：跨模型 heuristic 的确存在，尤其对 HeartMuLa/ACE-Step；但对未见 Suno 的泛化明显较弱，当前系统不能叫 universal AI-music detector。

## 五、vocal-active 与时长 sensitivity

raw-frozen vocal-active 数量也显示分轨行为差异：development/locked 中 Human 为 249/63，Suno 268/69，ACE-Step 233/55，而 HeartMuLa 为 400/100。HeartMuLa 的所有样本都超过 vocal/mix RMS 阈值，本身就是很强的 domain cue，可能同时来自生成器编配与 Demucs 对该音色的响应。

HeartMuLa 有 42/500 条原生输出不足 30 秒，最短 15.12 秒；统一协议在尾部补零。排除这 42 条并对 Human 等量下采样后，locked 92+92 的 full-band AUC 仍为 1.000，所以 trailing silence 不足以解释饱和结果。但这是 post-generation sensitivity，不能消除 genre/mastering 偏差。

## 六、HeartMuLa 与 ACE-Step 的逐 prompt 配对结果

两模型共享 500 条文本条件，因此 paired difference 比各自对 Human 更能隔离生成器差异。locked corrected 中最强差异包括：

| Metric | HeartMuLa median | ACE-Step median | Paired H−A | Separation AUC |
|---|---:|---:|---:|---:|
| sibilance burst rate | 1.625 | 1.130 | +0.527 | 0.904 |
| HF crest | 14.07 | 16.82 | −2.826 | 0.857 |
| HF modulation share | 0.2465 | 0.1417 | +0.0943 | 0.770 |
| fakeprint periodicity | 0.0327 | 0.0531 | −0.0286 | 0.757 |
| HF tilt 5–16 kHz | −6.13 | −8.06 | +3.083 | 0.707 |

共享 Demucs correction 对 paired absolute map 的最大影响只有 all 0.247 dB、locked 0.271 dB，符合“相同线性频率响应在生成器间 dB 差中大致相消”的预期。

## 七、Demucs bias correction 的结论

校正没有实质改变分类排序：

- Suno full-band raw/corrected AUC：0.879 / 0.876。
- HeartMuLa：1.000 / 1.000。
- ACE-Step：0.979 / 0.979。

locked average-map 的最大 raw→corrected 改变量约为 Suno 0.260 dB、HeartMuLa 0.274 dB、ACE-Step 0.285 dB。因而当前差异不是由一个共享的 Demucs average frequency response 主导；仍不能排除 class-dependent separation error。

## 八、必须写进 thesis 的偏差边界

Muse 音频从未下载或分析，只使用文本 metadata；不存在 Suno waveform leakage。但 500 条 prompt 明显偏 Pop/Ballad/Female Vocal，而 Human FMA 500 以 Rock/Electronic 为主，且 Human 与生成歌曲没有逐曲匹配。当前 AUC 可以同时包含：

- 生成器/codec 的真实频谱 fingerprint；
- genre、语言、人声比例和编配差异；
- mastering、响度和源数据 provenance；
- Demucs 对不同域的 class-dependent error。

论文应表述为“当前冻结域上的 discriminative spectral evidence”。下一步最关键的 ablation 是构造 style/language/vocal-activity matched Human cohort，再增加完全外部的人类与未知生成器测试集。

## 九、过程与产物

- 全部模型推理和 Demucs 只在 `5090-4` 的 RTX 5090 上执行；本地 UVR 未运行。
- 详细命令：`REPRODUCE_COMMANDS.md`。
- 版本、失败和运行记录：`RUN_LOG.md`、`MODEL_DATASET_RESEARCH.md`、`EXPERIMENT_PROTOCOL.md`。
- 主要结果：`analysis/comparison/`、`analysis/cross_generator/`、`analysis/cross_generator_vocal_active/`、`analysis/paired_generators/`、`analysis/human_vs_heartmula/`、`analysis/human_vs_acestep/`。
- 远端完整 raw/standardized audio、Demucs stems、JSONL state、logs、weights hashes 均保留在 `/mnt/nfs-code/users/yi/open_models_spectral_500_20260901`。

## 参考来源

- HeartMuLa official code: https://github.com/HeartMuLa/heartlib
- HeartMuLa paper: https://arxiv.org/abs/2601.10547
- ACE-Step 1.5 official code: https://github.com/ace-step/ACE-Step-1.5
- ACE-Step 1.5 paper: https://arxiv.org/abs/2602.00744
- Muse dataset: https://huggingface.co/datasets/bolshyC/Muse
- Muse paper: https://arxiv.org/abs/2601.03973
