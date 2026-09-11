# HeartMuLa / ACE-Step 500×2 频域实验协议

## 目标

在既有 Human vs Suno 500×500、Demucs vocal-only、频率偏差校正实验上，新增两个生成器域：

- HeartMuLa 500 条
- ACE-Step 1.5 500 条

两套生成器逐条共享同一个 500 条 prompt/lyrics 清单，并统一为 30 秒、44.1 kHz、双声道、16-bit FLAC。所有生成音频再使用同一 `htdemucs --two-stems=vocals` 管线，分别执行原始 vocal 频谱和减去冻结 Demucs average-difference spectrogram 后的校正频谱分析。

## 冻结来源与版本

- Prompt/lyrics: `bolshyC/Muse`，revision `b1bf3bf906daab3a896e14f6dea58cc295848452`，MIT。
- HeartMuLa code: `HeartMuLa/heartlib@3783bdb8441f2c298b1e64c8651173aac200361c`，Apache-2.0。
- HeartMuLa weights: `HeartMuLa/HeartMuLa-oss-3B-happy-new-year@41f6fc68490e11dc43fdabaa6b5767946408c903`。
- HeartCodec weights: `HeartMuLa/HeartCodec-oss-20260123@f889dab0532cfa4bf459f2a3367eb6d346b8eeda`。
- HeartMuLa tokenizer/config bundle: `HeartMuLa/HeartMuLaGen@9906b2bcd4598772a32cad4aec0760170fe0d177`。
- ACE-Step code: `ace-step/ACE-Step-1.5@ca1e85fe9430179831e6bc6be790c332190a3866`，MIT。
- ACE-Step weights: `ACE-Step/Ace-Step1.5@19671f406d603126926c1b7e2adc169acbcade22`。
- Demucs bias: 复用既有、冻结的 `demucs_frequency_bias.npz`，不得在本次 500×2 数据上重新估计。

Muse 音频由 Suno V5 合成，但本实验只下载和使用文本 metadata 中的 style 与 AI 自动生成歌词，不下载或使用 Muse 音频。这样避免把 Suno 波形带入 HeartMuLa/ACE-Step 的频谱测量。

## 500 条清单

- 中文 250、英文 250。
- `style_sim >= 0.4`。
- 必须有至少 30 秒来源时长和足够的非空歌词。
- 只取来源时间轴前 30 秒会出现的 lyric sections，防止把完整三分钟歌词强行压入 30 秒输出。
- ACE caption 必须不超过 512 字符；lyrics 不超过 4096 字符。
- 在全部合格记录中按 salted SHA-256 排序，各语言取最小 250 个，选择不依赖 JSONL 文件顺序。
- 每个语言固定 200 development、50 locked test；总计 400/100。
- 每条的整数 seed 由来源 id 哈希确定，两模型使用相同 seed 数值。

## 推理参数

HeartMuLa:

- `HeartMuLa-oss-3B-happy-new-year` + `HeartCodec-oss-20260123`
- 30,000 ms，top-k 50，temperature 1.0，CFG 1.5
- HeartMuLa bf16，HeartCodec fp32
- 模型与 codec 分卡常驻，避免每条重新加载

ACE-Step:

- `acestep-v15-turbo`
- 30 s，8 diffusion steps，固定 seed
- `thinking=False` 且所有 CoT 重写关闭，直接使用冻结 caption/lyrics
- 关闭 ACE 内建 peak normalization，避免形成只属于 ACE 的幅度预处理
- 保留当前发布版本 Turbo 的其余默认生成行为，包括 DCW `mode=double`、low scaler `0.05`、high scaler `0.02`、Haar wavelet

## 运行与审计原则

- 主机：运行前按 GPU memory/utilization/process/load/disk 重新选择；本次初始选择为 5090-4。
- 独立根目录：`/mnt/nfs-code/users/yi/open_models_spectral_500_20260901`。
- 每个模型独立 Python 环境、冻结依赖列表、独立 raw 输出与 JSONL state log。
- 每条写入 seed、耗时、采样率、声道、时长、字节数与 SHA-256。
- 输出先写 partial 文件，验证后原子 rename；已有合法输出会跳过，可断点续跑。
- 失败不静默替换 prompt；同一 prompt 以同一 seed 重试，最终失败条目单独报告。
- 标准化只做重采样、声道统一、截断/补零和 PCM 编码，不做响度归一化或 EQ。
- 标准化实现使用 SoundFile 读写与 SciPy `resample_poly`，统一输出 44.1 kHz、双声道、精确 1,323,000 frames、PCM-16 FLAC；重采样产生的超界样本会计数后 clip，而不会整曲 peak-normalize。
- ACE 环境的第一次标准化尝试因 TorchCodec/FFmpeg shared-library 不匹配而 500/500 失败；失败 JSONL 和日志单独保留，随后才切换到上述 SoundFile 实现。失败记录不覆盖成功记录。
- HeartMuLa smoke 通过后，正式清单先按不重叠索引分给三个双卡 worker；完成 158 条后再按实际缺失清单等量重排到四个双卡 worker。每次重排前先停止旧进程，重新划定互斥区间；已有合法输出做 resume skip，未完成 partial 删除后以同一 seed 重试，各次调度使用独立 state/log。
- Demucs 使用只读复用的已验证运行时（Demucs 4.0.1、PyTorch/Torchaudio 2.7.1+cu128），本实验不修改该环境；运行时完整 `pip freeze` 保存在本实验 logs。Demucs smoke 后正式分轨仅在 GPU 3 执行。
- 若生成器在 30 秒前原生结束，保留该行为并只在标准化阶段尾部补零，不重新生成或更换 prompt；分析时把 trailing-silence shortcut 作为显式限制，并以 raw-frozen vocal-active sensitivity subset 复核。

## 分析与可报告结论边界

1. 分别比较 Human vs Suno、Human vs HeartMuLa、Human vs ACE-Step；同时报告 pooled AI，但 pooled evaluation 只放入一份 Human 样本。
2. HeartMuLa vs ACE-Step 额外做 prompt-paired metric difference 和 average difference spectrogram，这是生成器间最干净的比较。
3. 为三个 Human-vs-generator 比较分别输出 all/development/locked-test 的四联 average difference spectrogram：raw/corrected × absolute/frame-normalized shape；再报告 full-band、0.3–5 kHz、5–10 kHz、5–16 kHz、去掉 5–10 kHz等既有 ablation。
4. 复用冻结校正曲线；若校正后提升很小，结论仍应是 Demucs artifact 不是主信号，而不是声称校正“失败”。
5. Muse 风格与既有人类集并未逐曲匹配，因此 Human-vs-generator AUC 仍可能含风格/制作域偏差；HeartMuLa-vs-ACE-Step 的 paired 结果才具有 prompt 控制。
6. 增加 single-generator train/test matrix、leave-one-generator-out 与 pooled-all 训练，并在全量 cohort 和由 raw vocal/mix activity 冻结的 vocal-active sensitivity subset 上各运行一次；这些 locked-test AUC 用来区分“可迁移 heuristic”和“单一生成器 fingerprint”。多生成器训练时只把 Human development 行作为回归权重重复，locked evaluation 中 Human 始终只出现一次。
7. 冻结 prompt cohort 以 Pop/Ballad/Female Vocal 为主，而 Human FMA cohort 以 Rock/Electronic 为主；因此任何 Human-vs-AI 分类结果都必须和 vocal-active sensitivity、跨生成器迁移及未来 style-matched Human ablation 一起解释，不能单独当作生成机制证据。
8. HeartMuLa 出现 42 条原生时长不足 30 秒；附加 post-generation sensitivity 会排除这些条目，并在 development/locked split 内用 salted SHA-256 等量下采样 Human。若 AUC 仍高，只能说明尾部补零不足以解释结果，不能消除风格和制作域偏差。
