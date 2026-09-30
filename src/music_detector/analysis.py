"""Numerical artifact measurements, explanations and separately versioned scoring."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import load_audio
from .artifacts import CPU_FAMILIES, DETAILS
from .scoring import FAMILIES
from .explanations import explain_family, metric_unit
from .algorithms import phase_features, musical_features, stereo_candidate_v1
from .algorithms import bicoherence_audio_v1, bicoherence_scalar_v1

ASSETS = Path(__file__).parent / 'assets'
DEPLOYMENT_BUNDLE = ASSETS / 'deployment_models.json'


def _clean_number(value):
    if value is None:
        return None
    value = float(value)
    if math.isinf(value):
        raise ValueError('Infinite measurement is an error, not scientific missingness.')
    return value if math.isfinite(value) else None


def analyze(path: Path, families: list[str], *, display_name: str | None = None,
            research: bool = False, progress: Callable[[str], None] = lambda stage: None,
            approved_region: dict | None = None, device: str | None = None) -> dict:
    if not families or len(families) != len(set(families)) or not set(families) <= set(FAMILIES):
        raise ValueError('Choose a nonempty set of known artifact families without duplicates.')
    selected = [key for key in FAMILIES if key in families]
    if 'BC' in selected and not research:
        raise ValueError('BC requires explicit research-only opt-in.')
    progress('校验音频与 Native30 预处理')
    view = load_audio(path, display_name=display_name, approved_region=approved_region)
    ac = view.stereo - view.stereo.mean(axis=0, keepdims=True)
    if float(np.max(np.abs(ac))) < 1e-8:
        raise ValueError('The selected 30-second region is silent or constant DC; no authorship probability is meaningful.')
    measurements = {}
    quality = {}
    neural_selected = [key for key in selected if key not in CPU_FAMILIES]
    if neural_selected:
        from .neural import extract_sdrp
        result = extract_sdrp(view, neural_selected,
                             device=device or os.environ.get('MUSIC_DETECTOR_DEVICE', 'cpu'), progress=progress)
        measurements.update({name: _clean_number(value) for name, value in result['features'].items()})
        quality.update(result['quality'])
        view.provenance['neural'] = result['provenance']
        view.provenance['neural_measurement_metadata'] = result['metadata']
    for family in selected:
        if family in neural_selected:
            continue
        progress('提取 ' + DETAILS[family][0])
        if family == 'F':
            raw = phase_features.extract_phase_features(view.mono_fh, 16000)
            quality[family] = raw['F_status']
        elif family == 'H':
            raw = musical_features.extract_musical_features(view.mono_fh, 16000)
            quality[family] = raw['H_status']
        elif family == 'SC':
            result = stereo_candidate_v1.extract_stereo_candidate(view.stereo, 44100, native_channels=2)
            raw = result['features']
            quality[family] = result['status']
        elif family == 'BC':
            crop = np.asarray(view.mono_bc[176000:304000] * 0.25, dtype=np.float64)
            result = bicoherence_audio_v1.extract(crop, 16000)
            reduction = bicoherence_scalar_v1.reduce_metadata(result['metadata'], crop={
                'start_sample': 176000, 'stop_sample_exclusive': 304000, 'source_resampled_samples': 480000})
            raw = {FAMILIES['BC'][0]: reduction['median_squared_bicoherence']}
            quality[family] = 'research_only'
        measurements.update({name: _clean_number(raw[name]) for name in FAMILIES[family]})

    warnings = ['这是参考数据分布下的相关性分析，不是音乐作者身份、侵权或使用 AI 的证明。',
                '音色、录音、编解码、混音与母带处理都可能改变这些指标。']
    if neural_selected:
        warnings.append('神经分析使用原算法与固定权重，但 Demucs 随机位移及 CPU/CUDA 精度可能改变特征；当前概率模型未单独完成新推理平台的外部校准验证。')
        if 'P' in neural_selected:
            warnings.append('All-In-One 分段权重为 CC-BY-NC-SA-4.0；本工具按研究/非商业用途集成，不授予新的权重使用许可。')
        vocal = view.provenance['neural_measurement_metadata'].get('vocal_activity', {})
        if 'S' in neural_selected and vocal.get('vocal_analysis_eligible') == 0:
            warnings.append('分离出的人声活动不足；S 按历史协议仍计算，但其频谱可能主要反映分离残留，不能解释为清晰人声的证据。')
    missing = [name for name, value in measurements.items() if value is None]
    if missing:
        warnings.append(f'{len(missing)} 个指标无法测得；缺失值保留为 null，不伪装成零。')
    prediction = dict(probability=None, raw_score=None, threshold=None, decision='unavailable',
                      calibration_status='model_not_installed', scope='尚无经过验收的概率模型；只展示实际测量。')
    scored = None
    if DEPLOYMENT_BUNDLE.is_file():
        progress('执行独立校准评分')
        from .calibration import predict_deployment
        predictive = [key for key in selected if key != 'BC']
        if predictive and any(measurements[name] is not None for key in predictive for name in FAMILIES[key]):
            bundle = json.loads(DEPLOYMENT_BUNDLE.read_text())
            values = {name: measurements[name] for key in predictive for name in FAMILIES[key]}
            scored = predict_deployment(bundle, values, predictive)
            prediction.update(scored)
        elif predictive:
            prediction.update(decision='abstain', calibration_status='abstained_no_observed_predictive_features',
                              scope='所选判别指标全部缺失；仅凭缺失模式不能对该音频给出 AI 概率。')
            warnings.append('没有可观测的判别指标，已拒绝给出概率；这不是 human 判定。')
        if 'BC' in selected:
            warnings.append('BC 仅显示研究测量，不参与概率或判定。')
    cards = []
    for family in selected:
        values = {name: measurements[name] for name in FAMILIES[family]}
        status = quality[family]
        if any(value is None for value in values.values()):
            status = 'partial' if any(value is not None for value in values.values()) else 'unavailable'
        cards.append(dict(id=family, name=DETAILS[family][0], status=status,
                          metrics=[dict(name=name, value=value, unit=metric_unit(name)) for name, value in values.items()],
                          **explain_family(family, values, scored)))
    return dict(audio=view.metadata, selected_artifacts=selected, features=cards,
                prediction=prediction, warnings=warnings, provenance=view.provenance,
                feature_values=measurements)
