"""Independent scalar replay for historical equal60 models; never fits parameters.

This numerical kernel is not a model/data admission gate. Callers must verify
the frozen model publication and measurement provenance before using it.
"""
import math


def score(row, model):
    columns = model['columns']
    n = len(columns)
    if not n or len(set(columns)) != n:
        raise ValueError('Empty or duplicate model columns')
    if model['threshold'] != 0.5 or model['feature_mode'] != 'values_plus_missing':
        raise ValueError('Unexpected historical model operating point')
    for key, size in [('medians', n), ('mean', 2*n), ('scale', 2*n),
                      ('coefficients_with_intercept', 2*n+1)]:
        if len(model[key]) != size or not all(math.isfinite(v) for v in model[key]):
            raise ValueError('Invalid frozen parameter: '+key)
    if any(v <= 0 for v in model['scale']):
        raise ValueError('Nonpositive frozen scale')
    values, missing = [], []
    for column, median in zip(columns, model['medians']):
        value = row[column]  # An absent column is an error, not missing data.
        if value is None or (isinstance(value, float) and math.isnan(value)):
            values.append(median)
            missing.append(1.0)
        else:
            if not math.isfinite(value):
                raise ValueError('Infinite measured feature')
            values.append(float(value))
            missing.append(0.0)
    beta = model['coefficients_with_intercept']
    terms = [beta[0]]
    for value, mean, scale, weight in zip(values+missing, model['mean'], model['scale'], beta[1:]):
        terms.append(((value-mean)/scale)*weight)
    raw = math.fsum(terms)
    if not math.isfinite(raw):
        raise ValueError('Nonfinite score')
    return raw, int(raw >= model['threshold'])


def positive_summary(predictions):
    """Single-class endpoint only; no invented human specificity or ROC AUC."""
    labels = list(predictions)
    if not labels or any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError('Expected nonempty binary predictions')
    tp = sum(labels)
    return dict(rows=len(labels), tp=tp, fn=len(labels)-tp,
                ai_sensitivity=tp/len(labels), false_negative_rate=1-tp/len(labels),
                human_specificity=None, balanced_accuracy=None, roc_auc=None)
