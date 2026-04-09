"""
Generate summary plots for run3-delta experiments.
Run from project root: python summary/run3-delta/generate_summary.py
"""
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = 'results'
OUT_DIR = 'summary/run3-delta'

EXPERIMENTS = {
    # name: (exp_dir, label, color, linestyle)
    'dg_disfa_f1': ('delta_graph_disfa_f1_fold1', 'ΔGraph+DISFA f1', '#1f77b4', '-'),
    'dg_disfa_f2': ('delta_graph_disfa_f2_fold1', 'ΔGraph+DISFA f2', '#ff7f0e', '-'),
    'dg_disfa_f3': ('delta_graph_disfa_f3_fold1', 'ΔGraph+DISFA f3', '#2ca02c', '-'),
    'dg_no_pt':    ('delta_graph_no_pretrain_fold1', 'ΔGraph (no pt)', '#d62728', '-'),
    'pe_disfa_f1': ('pe_delta_disfa_f1_fold1', 'PE+DISFA f1', '#1f77b4', '--'),
    'pe_disfa_f2': ('pe_delta_disfa_f2_fold1', 'PE+DISFA f2', '#ff7f0e', '--'),
    'pe_disfa_f3': ('pe_delta_disfa_f3_fold1', 'PE+DISFA f3', '#2ca02c', '--'),
    'pe_no_pt':    ('pe_delta_no_pretrain_fold1', 'PE (no pt)', '#d62728', '--'),
}


def parse_train_log(log_path):
    """Parse train.log and return list of dicts with epoch metrics."""
    epochs = []
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r'Epoch:\s+(\d+)\s+train_loss:\s+([\d.]+)\s+val_loss:\s+([\d.]+)\s+'
                r'val_mean_f1:\s+([\d.]+)\s+val_mean_acc:\s+([\d.]+)', line)
            if m:
                epochs.append({
                    'epoch': int(m.group(1)),
                    'train_loss': float(m.group(2)),
                    'val_loss': float(m.group(3)),
                    'val_f1': float(m.group(4)),
                    'val_acc': float(m.group(5)),
                })
    return epochs


def find_log(exp_dir):
    subdir = os.path.join(RESULTS_DIR, exp_dir)
    for d in os.listdir(subdir):
        log = os.path.join(subdir, d, 'train.log')
        if os.path.exists(log):
            return log
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_data = {}
    for key, (exp_dir, label, color, ls) in EXPERIMENTS.items():
        log = find_log(exp_dir)
        if log:
            all_data[key] = parse_train_log(log)

    # --- Plot 1: Training curves (F1) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ΔGraph experiments
    ax = axes[0]
    for key in ['dg_disfa_f1', 'dg_disfa_f2', 'dg_disfa_f3', 'dg_no_pt']:
        if key in all_data:
            d = all_data[key]
            _, label, color, ls = EXPERIMENTS[key]
            ax.plot([e['epoch'] for e in d], [e['val_f1'] for e in d],
                    label=label, color=color, linestyle=ls, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Mean F1 (%)')
    ax.set_title('Node-Level ΔGraph')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # PE-score experiments
    ax = axes[1]
    for key in ['pe_disfa_f1', 'pe_disfa_f2', 'pe_disfa_f3', 'pe_no_pt']:
        if key in all_data:
            d = all_data[key]
            _, label, color, ls = EXPERIMENTS[key]
            ax.plot([e['epoch'] for e in d], [e['val_f1'] for e in d],
                    label=label, color=color, linestyle=ls, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Mean F1 (%)')
    ax.set_title('Graph-Level PE-Score Delta')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Delta Model Training Curves — UNBC 3-Class', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'training_curves.png'), dpi=150)
    plt.close()

    # --- Plot 2: Best F1 comparison bar chart ---
    delta_names = ['ΔGraph\nDISFA f1', 'ΔGraph\nDISFA f2', 'ΔGraph\nDISFA f3', 'ΔGraph\n(no pt)']
    pe_names = ['PE-score\nDISFA f1', 'PE-score\nDISFA f2', 'PE-score\nDISFA f3', 'PE-score\n(no pt)']
    delta_keys = ['dg_disfa_f1', 'dg_disfa_f2', 'dg_disfa_f3', 'dg_no_pt']
    pe_keys = ['pe_disfa_f1', 'pe_disfa_f2', 'pe_disfa_f3', 'pe_no_pt']

    delta_f1 = [max(e['val_f1'] for e in all_data[k]) if k in all_data else 0 for k in delta_keys]
    pe_f1 = [max(e['val_f1'] for e in all_data[k]) if k in all_data else 0 for k in pe_keys]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(4)
    w = 0.35
    bars1 = ax.bar(x - w/2, delta_f1, w, label='Node-Level ΔGraph', color='#4c72b0')
    bars2 = ax.bar(x + w/2, pe_f1, w, label='PE-Score Delta', color='#dd8452')

    # Add baseline reference line
    ax.axhline(y=62.74, color='red', linestyle='--', alpha=0.7, label='FullPicture best (62.74%)')
    ax.axhline(y=53.71, color='gray', linestyle=':', alpha=0.7, label='FullPicture no pt (53.71%)')

    ax.set_ylabel('Best Val Mean F1 (%)')
    ax.set_title('Delta vs FullPicture — Best F1 Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['DISFA f1', 'DISFA f2', 'DISFA f3', 'No pretrain'])
    ax.legend()
    ax.set_ylim(0, 70)
    ax.grid(True, axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'best_metrics_comparison.png'), dpi=150)
    plt.close()

    # --- Plot 3: Per-class F1 at best epoch ---
    # Data extracted from logs
    per_class_data = {
        'ΔGraph+f1':  [61.5, 26.6, 5.3],
        'ΔGraph+f2':  [64.8, 23.6, 7.7],
        'ΔGraph+f3':  [54.5, 35.2, 3.4],
        'ΔGraph(np)': [52.2, 29.9, 5.1],
        'PE+f1':      [54.1, 23.5, 4.9],
        'PE+f2':      [58.5, 27.7, 3.0],
        'PE+f3':      [67.7, 31.3, 5.0],
        'PE(np)':     [66.2, 31.8, 4.8],
    }

    baseline_per_class = {
        'Full+DISFA f3': [89.8, 47.0, 51.4],
        'Full(no pt)':   [80.7, 37.4, 43.1],
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    all_labels = list(per_class_data.keys()) + list(baseline_per_class.keys())
    all_vals = list(per_class_data.values()) + list(baseline_per_class.values())

    x = np.arange(len(all_labels))
    w = 0.25
    no_pain = [v[0] for v in all_vals]
    mild = [v[1] for v in all_vals]
    pain = [v[2] for v in all_vals]

    ax.bar(x - w, no_pain, w, label='No Pain', color='#66c2a5')
    ax.bar(x, mild, w, label='Mild Pain', color='#fc8d62')
    ax.bar(x + w, pain, w, label='Pain', color='#8da0cb')

    ax.set_ylabel('F1 (%)')
    ax.set_title('Per-Class F1 at Best Epoch — Delta vs FullPicture Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=30, ha='right', fontsize=9)
    ax.axvline(x=7.5, color='black', linestyle='--', alpha=0.5)
    ax.text(3.5, 92, 'Delta Models', ha='center', fontsize=10, style='italic')
    ax.text(8.5, 92, 'Baselines', ha='center', fontsize=10, style='italic')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'per_class_f1.png'), dpi=150)
    plt.close()

    print(f'Plots saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
