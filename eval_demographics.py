"""
Demographic Bias Evaluation
============================
Computes per-subgroup AUROC, F1, and balanced accuracy, plus bias gap metrics.

Requires: scikit-learn  (pip install scikit-learn)

Usage (standalone):
    results = evaluate_by_demographics(
        all_labels, all_scores, all_preds,
        all_demographics={'gender': [...], 'ethnicity': [...], ...},
        output_dir='results/my_exp',
        prefix='epoch5_',
    )

Output structure:
    {
      "gender": {
        "M": {"auroc": 0.82, "f1": 0.74, "balanced_acc": 0.78, "n_samples": 120},
        "F": {"auroc": 0.79, "f1": 0.70, "balanced_acc": 0.75, "n_samples": 110}
      },
      "gender_bias_gap_auroc":        0.03,
      "gender_worst_group_auroc":     0.79,
      "gender_bias_gap_f1":           0.04,
      "gender_worst_group_f1":        0.70,
      "gender_bias_gap_balanced_acc": 0.03,
      "gender_worst_group_balanced_acc": 0.75,
      ...
    }
"""

import csv
import json
import os
from collections import defaultdict

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _check_sklearn():
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for demographic evaluation. "
            "Install with: pip install scikit-learn"
        )


def compute_group_metrics(y_true, y_score, y_pred):
    """
    Compute AUROC, F1, and balanced accuracy for one demographic subgroup.

    Args:
        y_true  (list/np.array): Binary ground-truth labels (0 or 1).
        y_score (list/np.array): Predicted probability of the positive class.
        y_pred  (list/np.array): Binary predicted labels (0 or 1).

    Returns:
        dict with keys: auroc, f1, balanced_acc, n_samples
    """
    _check_sklearn()
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred  = np.asarray(y_pred)

    result = {'n_samples': int(len(y_true))}

    # AUROC requires both classes to be present
    if len(np.unique(y_true)) > 1:
        result['auroc'] = float(roc_auc_score(y_true, y_score))
    else:
        result['auroc'] = float('nan')

    result['f1']           = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    result['balanced_acc'] = float(balanced_accuracy_score(y_true, y_pred))

    return result


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_by_demographics(
    all_labels,
    all_scores,
    all_preds,
    all_demographics,
    demo_keys=('age_group', 'gender', 'ethnicity'),
    output_dir=None,
    prefix='',
):
    """
    Compute per-subgroup metrics for each demographic attribute, then compute
    bias gaps and worst-group performance.

    Args:
        all_labels (list[int]):
            Ground-truth binary labels collected over the full validation set.
        all_scores (list[float]):
            Positive-class probability scores (used for AUROC).
        all_preds (list[int]):
            Binary hard predictions (argmax of model output).
        all_demographics (dict[str, list[str]]):
            Per-attribute group labels aligned with all_labels, e.g.
            {'gender': ['M', 'F', ...], 'ethnicity': ['White', ...], ...}
        demo_keys (tuple[str]):
            Which demographic attributes to evaluate (must be keys of all_demographics).
        output_dir (str|None):
            If provided, saves JSON and CSV results here.
        prefix (str):
            Optional filename prefix (e.g. 'epoch5_').

    Returns:
        results (dict):
            Nested dict of per-group metrics + bias gap summaries.
    """
    _check_sklearn()
    results = {}

    for attr in demo_keys:
        if attr not in all_demographics:
            continue

        group_labels = all_demographics[attr]  # parallel list to all_labels

        # Bucket samples by group value
        buckets = defaultdict(lambda: {'labels': [], 'scores': [], 'preds': []})
        for i, group_val in enumerate(group_labels):
            buckets[group_val]['labels'].append(all_labels[i])
            buckets[group_val]['scores'].append(all_scores[i])
            buckets[group_val]['preds'].append(all_preds[i])

        # Compute per-group metrics
        attr_metrics = {}
        for group_val, data in sorted(buckets.items()):
            attr_metrics[group_val] = compute_group_metrics(
                data['labels'], data['scores'], data['preds']
            )

        results[attr] = attr_metrics

        # Bias gap = max(group metric) − min(group metric)
        aurocs = [m['auroc'] for m in attr_metrics.values() if not np.isnan(m['auroc'])]
        f1s    = [m['f1']    for m in attr_metrics.values()]
        baccs  = [m['balanced_acc'] for m in attr_metrics.values()]

        results[f'{attr}_bias_gap_auroc']           = float(max(aurocs) - min(aurocs)) if len(aurocs) >= 2 else float('nan')
        results[f'{attr}_worst_group_auroc']        = float(min(aurocs))               if aurocs        else float('nan')
        results[f'{attr}_bias_gap_f1']              = float(max(f1s)    - min(f1s))
        results[f'{attr}_worst_group_f1']           = float(min(f1s))
        results[f'{attr}_bias_gap_balanced_acc']    = float(max(baccs)  - min(baccs))
        results[f'{attr}_worst_group_balanced_acc'] = float(min(baccs))

    _print_results(results, demo_keys)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _save_json(results, output_dir, prefix)
        _save_csv(results, demo_keys, output_dir, prefix)

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_results(results, demo_keys):
    print("\n" + "=" * 60)
    print("  Demographic Bias Evaluation")
    print("=" * 60)
    for attr in demo_keys:
        if attr not in results:
            continue
        print(f"\n[{attr.upper()}]")
        for group, m in results[attr].items():
            n     = m.get('n_samples', '?')
            auroc = f"{m['auroc']:.4f}" if not np.isnan(m['auroc']) else 'N/A'
            print(
                f"  {group:<20s}  n={n:<6}  "
                f"AUROC={auroc}  F1={m['f1']:.4f}  BalAcc={m['balanced_acc']:.4f}"
            )
        gap  = results.get(f'{attr}_bias_gap_auroc',    float('nan'))
        worst = results.get(f'{attr}_worst_group_auroc', float('nan'))
        gap_s  = f"{gap:.4f}"  if not np.isnan(gap)  else 'N/A'
        worst_s = f"{worst:.4f}" if not np.isnan(worst) else 'N/A'
        print(f"  >> Bias gap (AUROC): {gap_s}  |  Worst-group AUROC: {worst_s}")
    print("=" * 60 + "\n")


def _nan_to_none(obj):
    """Recursively convert nan floats to None for JSON serialisation."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _save_json(results, output_dir, prefix):
    fname = os.path.join(output_dir, f"{prefix}demographics.json")
    with open(fname, 'w') as f:
        json.dump(_nan_to_none(results), f, indent=2)
    print(f"Saved demographic results → {fname}")


def _save_csv(results, demo_keys, output_dir, prefix):
    fname = os.path.join(output_dir, f"{prefix}demographics.csv")
    rows = []
    for attr in demo_keys:
        if attr not in results:
            continue
        for group, m in results[attr].items():
            rows.append({
                'attribute':    attr,
                'group':        group,
                'n_samples':    m.get('n_samples', ''),
                'auroc':        '' if np.isnan(m['auroc']) else m['auroc'],
                'f1':           m['f1'],
                'balanced_acc': m['balanced_acc'],
            })
    if rows:
        with open(fname, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['attribute', 'group', 'n_samples', 'auroc', 'f1', 'balanced_acc']
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved demographic CSV    → {fname}")
