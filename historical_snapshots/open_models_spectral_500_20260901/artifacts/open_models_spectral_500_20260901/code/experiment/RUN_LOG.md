# HeartMuLa / ACE-Step 500×2 实验运行记录

日期：2026-09-01（Asia/Shanghai；state 文件内部时间为 UTC）

远端实验根目录：`/mnt/nfs-code/users/yi/open_models_spectral_500_20260901`

本地轻量归档：`/Users/yi/Documents/code/music/artifacts/open_models_spectral_500_20260901`

## 1. 设计冻结

- 复用既有 Human 500 / Suno 500、30 秒、44.1 kHz、双声道、lossless FLAC cohort 和同一个 development 400 / locked test 100 划分。
- 新增 HeartMuLa 500 与 ACE-Step 500；两模型共享相同 500 条 Muse 文本条件、整数 seed、语言比例和 split。
- Muse 只读取 metadata 文本。配套 Suno V5 音频没有下载、解码或进入任何频谱计算。
- 冻结 prompt manifest：`prompts/frozen_500/prompt_manifest.jsonl`，SHA-256 `bb0941b91d41d10a978cb9e4b077399e89fea92526ee469e0e6743b09efb917b`。
- manifest 为中文 250、英文 250；development 400、locked test 100；500 个唯一 id 和 500 个唯一 seed。独立重建 hash 一致。
- 冻结 Demucs 校正谱：`analysis/demucs_frequency_bias.npz`，SHA-256 `bcacfecac5ce69927dfed2a51ec21cc346f613a7874c519e419d22d35a97de5e`。

## 2. 数据与版本

- Muse revision：`b1bf3bf906daab3a896e14f6dea58cc295848452`。
- 原始中文 metadata：49,516 行，SHA-256 `e00f8ff86cc993e0e6b8ed39a3707772edef43a85d49bcb33229dc2cb9d69c67`。
- 原始英文 metadata：66,621 行，SHA-256 `59a6efc096af5dbd17f5669ed4f78a1d272e8c3ddb5d507795ac2f1917d803f2`。
- HeartMuLa code：`3783bdb8441f2c298b1e64c8651173aac200361c`。
- HeartMuLa / HeartCodec / tokenizer revisions：`41f6fc...c903` / `f889da...eeda` / `9906b2...177`；逐文件权重 hash 保存在 `logs/heartmula_weight_sha256.txt`。
- ACE-Step code：`ca1e85fe9430179831e6bc6be790c332190a3866`。
- ACE-Step weights revision：`19671f406d603126926c1b7e2adc169acbcade22`；逐文件 hash 保存在 `logs/acestep_weight_sha256.txt`。
- ACE 官方 handler 初始化时把 checkpoint 内的 `configuration_acestep_v15.py` 与 `modeling_acestep_v15_turbo.py` 同步为当前固定代码版本；模型权重未改变。收尾已对 87 个 checkpoint 文件重算最终 hash，清单为 `logs/acestep_weight_sha256_final.txt`，清单自身 SHA-256 `9eacb4c09e0f1887f257e333dd07e368612705def13812e3b5bfe6f16cda2bc9`；下载后/handler 前清单也保留。

## 3. 5090 选择与环境

- 独立检查 5090-2、5090-4、5090-5 后选择 5090-4；开始时 8 张 RTX 5090 都为 2 MiB、0% utilization，host load 约 0.07。
- 5090-5 的 GPU 0–3 当时占用 22–27 GB、load 约 11，未使用。
- ACE 独立环境：Python 3.12.13、PyTorch 2.10.0+cu128；freeze 为 `logs/acestep_pip_freeze.txt`。
- HeartMuLa 独立环境：Python 3.10、PyTorch 2.7.1+cu128；freeze 为 `logs/heartmula_pip_freeze.txt`。
- Demucs 使用只读复用、已验证的环境：Demucs 4.0.1、PyTorch/Torchaudio 2.7.1+cu128；没有修改该环境，freeze 复制到 `logs/demucs_runtime_pip_freeze.txt`。

## 4. Smoke test

### ACE-Step

- GPU：2；thinking/CoT 关闭；Turbo 8 steps；内建 normalization 关闭；发布默认 DCW 保留。
- smoke 输出精确 30.0 秒、48 kHz、双声道 PCM-16 FLAC，SHA-256 `dfb750eed17b160a8fe7b62a277a60767966a6b1765fbb733e4068dd385466ab`。
- cold init 约 32 秒；首条约 6.3 秒；正式批次稳态约 0.95–1.0 秒/条；峰值显存日志约 7.29 GB。

### HeartMuLa

- GPU 0：HeartMuLa bf16；GPU 1：HeartCodec fp32；top-k 50、temperature 1.0、CFG 1.5。
- smoke model init 29.52 秒，生成 29.43 秒；原始输出 30.08 秒、48 kHz、双声道 float WAV，SHA-256 `09b6c489fdbc96fa8dd39eb4ecd685a321cd6adabc7eb4748a72c2c131da1328`。

### Demucs

- GPU 3；`htdemucs --two-stems vocals --float32 --shifts 0 --overlap 0.25 --segment 7`。
- 一条 ACE 标准化样本 smoke 通过，得到两个精确 30 秒 float32 WAV stem。

## 5. ACE-Step 正式生成与标准化

- 正式生成：UTC 08:12:22 初始化，UTC 08:20:31 完成最后一条。
- 验证：500 manifest、500 generated events、500 ok、0 error、500 unique ids、500 文件；全部 48 kHz、双声道、精确 30 秒、finite。
- 原始 peak 中位数 0.999969，范围 0.757812–1.0；RMS 中位数 0.14306，范围 0.06518–0.24580。这里没有执行额外 peak normalization。
- 第一次标准化使用 Torchaudio 时，因 TorchCodec 找不到兼容 FFmpeg shared libraries 导致 500/500 失败。失败 state 保留为 `state/acestep_standardization_failed_torchcodec.jsonl`，失败日志为 `logs/acestep_standardization.log`。
- 修正后改用 SoundFile + `scipy.signal.resample_poly`；500/500 输出为 44.1 kHz、双声道、1,323,000 frames、30.0 秒、PCM-16 FLAC。
- 多相重采样后共有 259 首出现至少一个绝对值略超 1 的 sample，但全批次合计仅 349 个 sample；按冻结实现逐样本 clip，未整曲归一化。

## 6. ACE-Step Demucs

- 正式运行：UTC 08:30:59–08:48:55，总计 1,075.19 秒。
- 25 个 batch × 20 tracks；最终验证 500 tracks、1,000 stems，全为 44.1 kHz、双声道、精确 1,323,000 frames、float32 WAV。
- 每 batch 的首末文件、耗时和累计验证数保存在 `state/demucs_acestep.jsonl`。

## 7. HeartMuLa 正式生成

- smoke 通过后先用三个不重叠双卡 worker：GPU 0–1 负责 0–166；GPU 4–5 负责 167–333；GPU 6–7 负责 334–499。
- 为避免原先 GPU 0–1 worker 越过 166 与第二路发生竞争，曾在一条样本写出前主动中断并按 `--limit 167` 重启；该条留下一个 `BrokenPipeError` 历史 event，partial 已删除并由相同 seed 重试成功。
- 完成 158 条后，为利用空闲 GPU 2–3 并缩短等待时间，安全停止三路 worker，按当时实际缺失样本等量重排为四路：49–134、135–277、278–414、415–499。区间包含的既有合法文件由 resume 校验跳过；四路各含 85–86 个真正缺失样本，分别使用 GPU 0–1、2–3、4–5、6–7。
- 每次调度均使用独立 JSONL state 与日志；正式区间互斥，逐条使用 partial + atomic rename。旧 state/log 和人为中断留下的历史错误均保留，最终有效状态按每个 id 的最后一次 event 判定。
- 最终生成文件 500、唯一有效 id 500、partial 0；共有 504 个 generated events，其中 4 个为受控中断触发的历史 `BrokenPipeError`，均以同 prompt/seed 重试并恢复，最终有效 error id 为 0。只读 reconciliation 发现全部 500 id 已有 ok state，因此实际补写 0 条。
- 第一条正式样本开始于 UTC 08:37:02，最后一条完成于 UTC 09:46:51；包含受控中断、重新加载和重平衡在内约 69 分 50 秒。单条推理耗时中位数 23.34 秒，范围 11.74–40.96 秒。
- 原始输出时长中位数/最大值均为 30.08 秒，但 42 条由模型提前结束，最短 15.12 秒。第一次要求原始输出至少 30 秒的严格 validation 因这 42 条失败，报告保留为 `reports/heartmula_generation_validation_raw_strict_failed.json`；允许模型原生长度 5–30.2 秒的完整性 validation 随后通过。
- 原始 peak 中位数 0.90754、范围 0.40235–2.03729；RMS 中位数 0.14489、范围 0.04474–0.26792。

## 7.1 HeartMuLa 标准化

- 500/500 成功，全部 44.1 kHz、双声道、1,323,000 frames、PCM-16 FLAC，500 个唯一输出 hash。
- 42 条短输出尾部补零，总 padding 11,501,280 frames，单条最多 656,208 frames；这会造成显著 trailing silence，属于模型提前结束与统一长度协议的交互，可能成为分类捷径。
- 457 条 30.08 秒输出被截断，单条最多 3,528 frames（约 80 ms），总截断 1,612,296 frames。
- 重采样后 164 首有至少一个超出 `[-1,1]` 的 sample，总计 5,085、单首最多 457；逐样本 clip，未做整曲 normalization。

## 8. 分析产物

- Human vs ACE-Step：`analysis/human_vs_acestep/`。
- Human vs HeartMuLa：`analysis/human_vs_heartmula/`。
- Human vs Suno average maps：`analysis/human_vs_suno_average/`。
- Prompt-paired HeartMuLa vs ACE-Step：`analysis/paired_generators/`。
- 三生成器汇总：`analysis/comparison/`。
- Cross-generator / leave-one-generator-out：`analysis/cross_generator/`。
- 每个 Human-vs-generator 比较输出 all/development/locked-test 四联热图：raw/corrected × absolute/frame-normalized spectral shape。

ACE-Step 当前 locked-test 结果已完成动态来源名修正并通过数值有限性检查：full-band corrected AUC 0.979，5–10 kHz only 0.971，5–16 kHz only 0.984，without 5–10 kHz 0.988，0.3–5 kHz 0.931。raw/corrected 变化很小，初步支持冻结 Demucs average-response 不是 ACE-Step 区分度的主要来源；最终解释需等待 HeartMuLa 与 cross-generator 对照。

## 9. HeartMuLa Demucs 与正式分析结果

- HeartMuLa Demucs：UTC 09:54:26–10:04:21，594.36 秒；500 tracks、1,000 stems，全为 44.1 kHz、双声道、1,323,000 frames、float32 WAV。
- HeartMuLa locked corrected AUC：full 1.000、5–10 kHz 0.9999、without 5–10 kHz 0.9999、5–16 kHz 0.9998、0.3–5 kHz 0.9819；全部 CSV 数值 finite。
- 排除 42 条原生短于 30 秒的 HeartMuLa 后，development 366+366、locked 92+92；full-band corrected AUC 仍为 1.000。
- Average shape map 的 locked AI−Human：HeartMuLa 10–16 kHz `+14.104 dB`、16–20 kHz `−15.912 dB`；ACE-Step 为 `+8.465 / +10.519 dB`；Suno 为 `+4.322 / +0.146 dB`。
- Leave-one-generator-out corrected best AUC：all tracks Suno 0.698、HeartMuLa 0.949、ACE-Step 0.834；vocal-active 为 0.702、0.942、0.935。
- Pooled-all locked best AUC：all tracks 0.943；vocal-active 0.969。单模型跨域矩阵存在明显不对称和低于 0.5 的方向反转，证明高单域 AUC 混有 generator fingerprint。
- Prompt-paired HeartMuLa-vs-ACE 最强 locked corrected differences：sibilance burst rate separation AUC 0.904、HF crest 0.857、HF modulation share 0.770、fakeprint periodicity 0.757。
- 冻结 Demucs correction 对 locked average map 的最大变化：HeartMuLa 0.274 dB、ACE-Step 0.285 dB；对 full-band AUC 影响近乎为零。

## 10. 最终产物状态

- 原始音频、标准化音频、stems、完整 logs/state/weights hashes 保留在远端 NFS；本地同步报告、CSV/PNG/JSON、代码、prompt manifest 和少量试听样本。
- 总结见 `THESIS_RESULTS_CN.md`；精确复现命令见 `REPRODUCE_COMMANDS.md`。
- 轻量关键文件的最终 hash 清单为 `reports/LIGHTWEIGHT_SHA256.txt`；清单自身 hash 在最终交付信息中记录，避免文档与其校验清单产生循环依赖。
