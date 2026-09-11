# 模型与 prompt 数据调研记录（2026-09-01）

## 最终选择

### HeartMuLa

- 论文：[HeartMuLa: A Family of Open Sourced Music Foundation Models](https://arxiv.org/abs/2601.10547)
- 官方代码：[HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib)
- 本实验代码提交：`3783bdb8441f2c298b1e64c8651173aac200361c`
- 官方在 2026-02-13 推荐 `HeartMuLa-oss-3B-happy-new-year` 与 `HeartCodec-oss-20260123` 组合。
- 模型权重 revision：`41f6fc68490e11dc43fdabaa6b5767946408c903`，约 15.75 GB，Apache-2.0。
- codec revision：`f889dab0532cfa4bf459f2a3367eb6d346b8eeda`，约 6.64 GB，Apache-2.0。
- tokenizer/config revision：`9906b2bcd4598772a32cad4aec0760170fe0d177`。
- 官方输入契约是 lyrics + comma-separated tags；默认/推荐推理关键参数为 top-k 50、temperature 1.0、CFG 1.5，HeartMuLa bf16、HeartCodec fp32。

### ACE-Step 1.5

- 论文：[ACE-Step 1.5](https://arxiv.org/abs/2602.00744)
- 官方代码：[ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5)
- 本实验代码提交：`ca1e85fe9430179831e6bc6be790c332190a3866`
- 权重：[ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5)，revision `19671f406d603126926c1b7e2adc169acbcade22`，约 10.08 GB，MIT。
- 官方 inference contract：caption 最长 512 字符、lyrics 最长 4096 字符、duration 10–600 秒；Turbo 推荐 8 steps，并支持固定 seed。
- 官方文档说明当输入 metadata 已精确且需要完整控制时可关闭 thinking；本实验关闭 thinking/CoT，防止模型重写冻结 prompt 与歌词。
- ACE 内建 peak normalization 会改变频域强度相关指标，因此本实验显式关闭；后续两模型统一进行同一 lossless standardization。

## Prompt 数据集筛选

### 未用作主清单：HeartMuLa-Benchmark

- 官方地址：[HeartMuLa-Benchmark](https://modelscope.cn/datasets/HeartMuLa/HeartMuLa-Benchmark)
- revision：`ab18c82248b7bc0ba5bda8acc76357516b221060`
- Apache-2.0，包含 AI-generated tags 与 lyrics。
- 实际只有 80 条：中文 20、英文 30、日/韩/西各 10，不足用户要求的 500 条。
- 因此保留为后续 external-condition subset，不通过重复或扰动伪扩增到 500。

### 主清单：Muse

- 数据页：[bolshyC/Muse](https://huggingface.co/datasets/bolshyC/Muse)
- 论文：[Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control](https://arxiv.org/abs/2601.03973)
- revision：`b1bf3bf906daab3a896e14f6dea58cc295848452`
- MIT，116,137 条中英文 metadata，包含 style、分段歌词、section timing、style similarity 与音频路径。
- ACE-Step 官方 large-scale SFT 文档直接给出了 Muse `style -> caption`、`sections[].text -> lyrics` 的映射，说明其字段与 ACE 输入契约兼容。
- Muse 配套音频由 Suno V5 合成；本实验只使用 metadata 文本条件，不下载、不读取、不分析这些音频。

## 为什么不用 MusicCaps/MusicBench 作为本次主清单

- MusicCaps 是约 5.5k 条 expert captions，MusicBench 扩展到约 52k 条控制 caption；它们适合 instrumental text-to-music。
- 两者没有与每条 caption 配对、许可明确且结构化的生成歌词。若强行让 HeartMuLa 使用 `[Instrumental]`，本实验最关心的 vocal 高频与 Demucs vocal-only 分析会退化成“vocal leakage”分析。
- Muse 同时提供 style 与 AI 自动生成的分段歌词，因此更贴合本次 vocal-focused 频域实验，同时可以给两模型完全相同的语义条件。

## 偏差边界

- Muse 的文本条件来自 Suno 合成语料，因此 prompt 风格可能偏向 Suno 的输入分布；这是 prompt-domain 偏差，不是波形泄漏。
- 冻结的 500 条 prompt 明显偏向主流人声歌曲：最常见标签为 Pop 319、Ballad 292、Female Vocal 286、Piano 263、C-pop 233；作为 Human 对照的 FMA 500 则以 Rock 163、Electronic 115、Experimental 45、Hip-Hop 42 为主，Pop 只有 21。因而 Human-vs-generator 的高 AUC 不能排除 genre/vocal-arrangement domain shift。
- HeartMuLa 与 ACE-Step 逐 prompt 配对可控制 prompt 组成，但它们与既有人类音乐集并未逐曲匹配；Human-vs-generator AUC 仍可能包含 genre、language、mastering 与数据来源偏差。
- 论文结论应以“这些指标在当前冻结域上提供区分信息”表述，不应宣称为普适 AI 音乐检测器。
