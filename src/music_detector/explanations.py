"""Measurement-aware explanations; no causal or universal authorship claims."""
from __future__ import annotations

import math


def metric_unit(name: str) -> str:
    if name.startswith('d__dynamics_'):
        return 'dB'
    if '_ms_' in name:
        return 'ms'
    if 'db_oct' in name:
        return 'dB/oct'
    if '_db' in name:
        return 'dB'
    if name == 'p__section_duration_median':
        return 's'
    if name == 'p__section_bars_median':
        return 'bars'
    if 'per_khz' in name:
        return 'peaks/kHz'
    if name.endswith('_hz'):
        return 'Hz'
    return ''


def explain_family(family: str, values: dict, scored: dict | None) -> dict:
    observed = sum(value is not None for value in values.values())
    reason = f'已测得 {observed}/{len(values)} 个指标。'
    if family == 'BC':
        return dict(reason=reason + '固定三元组的相位耦合描述，不参与概率或判定，也不能证明 AI 来源或非线性成因。',
                    contribution=None, value_contribution=None, missingness_contribution=None,
                    evidence=[])
    terms = [item for item in (scored or {}).get('contributions', []) if item['family'] == family]
    if not terms:
        return dict(reason=reason + '未获得有效模型贡献，不能单凭数值判断方向。',
                    contribution=None, value_contribution=None, missingness_contribution=None,
                    evidence=[])
    actual = math.fsum(item['value_contribution'] for item in terms)
    missingness = math.fsum(item['missingness_contribution'] for item in terms)
    total = actual + missingness
    direction = 'AI 标签' if total > 0 else 'human 标签' if total < 0 else '零净贡献'
    reason += f'相对训练集中心，该组对原始分数的贡献为 {total:+.4f}，方向为{direction}。'
    evidence = []
    for item in sorted(terms, key=lambda value: abs(value['value_contribution']), reverse=True)[:3]:
        name = item['feature']
        value = values[name]
        evidence.append(dict(feature=name, value=value, unit=metric_unit(name),
                             value_contribution=item['value_contribution'],
                             missingness_contribution=item['missingness_contribution'],
                             missing=item['missing']))
    strongest = next((item for item in evidence if item['value'] is not None), None)
    if strongest:
        reason += (f"其中 {strongest['feature']} 实测 {strongest['value']:.5g} {strongest['unit']}，"
                   f"数值项贡献 {strongest['value_contribution']:+.4f}。")
    reason += (f'缺失模式项单独贡献 {missingness:+.4f}；它反映测量可用性，不能解释为音乐的固有 AI 特征。'
               '贡献解释原始分数，不是概率百分点或因果关系。')
    return dict(reason=reason, contribution=total, value_contribution=actual,
                missingness_contribution=missingness, evidence=evidence)
