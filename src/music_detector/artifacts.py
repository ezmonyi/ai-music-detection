"""User-facing registry; availability describes real installed implementations."""
from .scoring import FAMILIES

DETAILS = {
    'S': ('高频人声频谱', 'Demucs 人声分轨，分离偏差校正后的高频谱与齿音统计。'),
    'D': ('伴奏动态', '非人声分轨的能量跨度、四分位距与相邻变化。'),
    'R': ('节奏变化', 'Beat This! 节拍间隔、速度变化与节奏熵。'),
    'P': ('乐句与段落', 'All-In-One 分段及小节长度变化。'),
    'F': ('相位演化', '相位残差与群延迟在起音、持续和衰减区间的统计。'),
    'H': ('音高类别路径', '色度熵与音高类别轨迹变化；不是和弦识别。'),
    'SC': ('立体声关系', '三个频带内的声道电平差与侧声道能量占比。'),
    'BC': ('双相干 · 研究项', '固定频率三元组的描述性双相干；不作为 AI 概率证据。'),
}
CPU_FAMILIES = frozenset({'F', 'H', 'SC', 'BC'})


def catalogue() -> list[dict]:
    from .neural import runtime_status
    records = []
    for key, (name, description) in DETAILS.items():
        ready, reason = key in CPU_FAMILIES, None
        if not ready:
            # Cheap availability probe; the execution path always verifies
            # every pinned source/weight hash before using the backend.
            status = runtime_status(families=[key], verify_hashes=False)
            ready = status['ready']
            if not ready:
                reason = '需要显式安装分析依赖和权重；不会使用替代指标。' + ' / '.join(status['errors'][:2])
        records.append(dict(id=key, name=name, description=description, available=ready,
                            research_only=key == 'BC', reason=reason, columns=FAMILIES[key]))
    return records
