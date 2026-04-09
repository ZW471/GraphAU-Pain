"""
Generate summary for run4 experiments:
  1) SynPAIN no-delta (Swin-B + ResNet-50)
  2) SynPAIN Part1/Part2 → UNBC finetuning
  3) SynPAIN demographic analysis without delta

Run: uv run python summary/run4-nodelta-parts/generate_summary.py
"""
import os, re, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUT_DIR = BASE_DIR

# ── Parse functions ─────────────────────────────────────────────────────────

def parse_synpain_log(log_path):
    epochs = []
    current_epoch = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Epoch\s+\[(\d+)/\d+\]', line)
            if m:
                current_epoch = int(m.group(1))
                continue
            m = re.search(r'train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+val_acc=([\d.]+)%?\s+val_f1=([\d.]+)%?', line)
            if m and current_epoch is not None:
                epochs.append({
                    'epoch': current_epoch,
                    'train_loss': float(m.group(1)),
                    'val_loss': float(m.group(2)),
                    'val_acc': float(m.group(3)),
                    'val_f1': float(m.group(4)),
                })
                current_epoch = None
    return epochs


def parse_unbc_log(log_path):
    epochs = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Epoch:\s+(\d+)\s+train_loss:\s+([\d.]+)\s+val_loss:\s+([\d.]+)\s+val_mean_f1_score\s+([\d.]+),val_mean_acc\s+([\d.]+)', line)
            if m:
                epochs.append({
                    'epoch': int(m.group(1)),
                    'train_loss': float(m.group(2)),
                    'val_loss': float(m.group(3)),
                    'val_f1': float(m.group(4)),
                    'val_acc': float(m.group(5)),
                })
    return epochs


def parse_unbc_per_class_3(log_path):
    results = []
    with open(log_path) as f:
        lines = f.readlines()
    expect_f1 = False
    for line in lines:
        if 'F1-score-list' in line:
            expect_f1 = True
            continue
        if expect_f1:
            m = re.search(r"'No Pain:\s+([\d.]+)\s+Mild Pain:\s+([\d.]+)\s+Pain:\s+([\d.]+)'", line)
            if m:
                results.append({
                    'no_pain': float(m.group(1)),
                    'mild_pain': float(m.group(2)),
                    'pain': float(m.group(3)),
                })
            expect_f1 = False
    return results


def deduplicate_epochs(epochs):
    seen = {}
    for e in epochs:
        seen[e['epoch']] = e
    return sorted(seen.values(), key=lambda e: e['epoch'])

# ── Experiment definitions ──────────────────────────────────────────────────

RUN_3CLASS = os.path.join(RESULTS_DIR, 'run-3class')

SYNPAIN_EXPS = {
    # Existing baselines for comparison
    'delta_swin': ('SynPAIN Delta (Swin-B)', os.path.join(RESULTS_DIR, 'run1', 'bs_32_seed_0_lr_1e-05')),
    'delta_r50': ('SynPAIN Delta (R50)', os.path.join(RUN_3CLASS, 'synpain_delta_r50', 'bs_32_seed_0_lr_1e-05')),
    # New no-delta experiments
    'nodelta_swin': ('SynPAIN NoDelta (Swin-B)', os.path.join(RESULTS_DIR, 'synpain_nodelta_swin', 'bs_32_seed_0_lr_1e-05')),
    'nodelta_r50': ('SynPAIN NoDelta (R50)', os.path.join(RESULTS_DIR, 'synpain_nodelta_r50', 'bs_32_seed_0_lr_1e-05')),
    # Part A/B experiments
    'part1_r50': ('SynPAIN Part1 (R50)', os.path.join(RESULTS_DIR, 'synpain_part1_r50', 'bs_32_seed_0_lr_1e-05')),
    'part2_r50': ('SynPAIN Part2 (R50)', os.path.join(RESULTS_DIR, 'synpain_part2_r50', 'bs_32_seed_0_lr_1e-05')),
}

UNBC_EXPS = {
    # Baselines from run2
    'unbc_baseline': ('UNBC Baseline', os.path.join(RUN_3CLASS, 'unbc_finetune_3class_fold1', 'bs_64_seed_0_lr_0.0001')),
    'unbc_disfa_f3': ('UNBC+DISFA f3', os.path.join(RUN_3CLASS, 'unbc_3class_from_disfa_f3_fold1', 'bs_64_seed_0_lr_0.0001')),
    'unbc_synpain_r50': ('UNBC+SynPAIN R50 (delta)', os.path.join(RUN_3CLASS, 'unbc_3class_from_synpain_r50_fold1', 'bs_64_seed_0_lr_0.0001')),
    # New UNBC experiments
    'unbc_nodelta_r50': ('UNBC+NoDelta R50', os.path.join(RESULTS_DIR, 'unbc_3class_from_nodelta_r50_fold1', 'bs_64_seed_0_lr_0.0001')),
    'unbc_part1_r50': ('UNBC+Part1 R50', os.path.join(RESULTS_DIR, 'unbc_3class_from_part1_r50_fold1', 'bs_64_seed_0_lr_0.0001')),
    'unbc_part2_r50': ('UNBC+Part2 R50', os.path.join(RESULTS_DIR, 'unbc_3class_from_part2_r50_fold1', 'bs_64_seed_0_lr_0.0001')),
}


def load_all():
    data = {}
    for key, (name, path) in SYNPAIN_EXPS.items():
        log = os.path.join(path, 'train.log')
        if os.path.exists(log):
            epochs = deduplicate_epochs(parse_synpain_log(log))
            data[key] = {'name': name, 'epochs': epochs, 'type': 'synpain', 'path': path}
            print(f'  {key}: {len(epochs)} epochs')
    for key, (name, path) in UNBC_EXPS.items():
        log = os.path.join(path, 'train.log')
        if os.path.exists(log):
            epochs = deduplicate_epochs(parse_unbc_log(log))
            pc = parse_unbc_per_class_3(log)
            if len(pc) > len(epochs):
                pc = pc[-len(epochs):]
            data[key] = {'name': name, 'epochs': epochs, 'type': 'unbc', 'per_class': pc, 'path': path}
            print(f'  {key}: {len(epochs)} epochs, {len(pc)} per-class')
    return data


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_synpain_comparison(data):
    """Compare delta vs no-delta SynPAIN pretraining."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('SynPAIN Pretraining: Delta vs No-Delta', fontsize=14, fontweight='bold')

    colors = {'delta_swin': '#1f77b4', 'delta_r50': '#ff7f0e',
              'nodelta_swin': '#2ca02c', 'nodelta_r50': '#d62728',
              'part1_r50': '#9467bd', 'part2_r50': '#8c564b'}

    # F1 curves
    ax = axes[0]
    for key in ['delta_swin', 'delta_r50', 'nodelta_swin', 'nodelta_r50', 'part1_r50', 'part2_r50']:
        if key in data:
            ep = data[key]['epochs']
            ax.plot([e['epoch'] for e in ep], [e['val_f1'] for e in ep],
                    marker='.', label=data[key]['name'], color=colors.get(key))
    ax.set_title('Validation F1 (%)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 (%)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Loss curves
    ax = axes[1]
    for key in ['delta_swin', 'delta_r50', 'nodelta_swin', 'nodelta_r50', 'part1_r50', 'part2_r50']:
        if key in data:
            ep = data[key]['epochs']
            ax.plot([e['epoch'] for e in ep], [e['val_loss'] for e in ep],
                    marker='.', label=data[key]['name'], color=colors.get(key))
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'synpain_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def plot_unbc_comparison(data):
    """Compare UNBC finetuning from different pretraining sources."""
    unbc_keys = [k for k in UNBC_EXPS if k in data and data[k]['epochs']]
    if not unbc_keys:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('UNBC 3-Class Finetuning from Different Pretraining', fontsize=14, fontweight='bold')

    for key in unbc_keys:
        ep = data[key]['epochs']
        axes[0].plot([e['epoch'] for e in ep], [e['val_f1'] for e in ep],
                     marker='.', label=data[key]['name'])
        axes[1].plot([e['epoch'] for e in ep], [e['val_loss'] for e in ep],
                     marker='.', label=data[key]['name'])

    axes[0].set_title('Val Mean F1 (%)'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('F1 (%)')
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
    axes[1].set_title('Val Loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'unbc_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def plot_best_metrics_bar(data):
    """Bar chart of best F1 across all experiments."""
    all_keys = list(SYNPAIN_EXPS.keys()) + list(UNBC_EXPS.keys())
    names, f1s, accs = [], [], []
    for key in all_keys:
        if key not in data or not data[key]['epochs']:
            continue
        best = max(data[key]['epochs'], key=lambda e: e['val_f1'])
        names.append(data[key]['name'])
        f1s.append(best['val_f1'])
        accs.append(best['val_acc'])

    if not names:
        return

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - w/2, f1s, w, label='Best F1 (%)', color='#4C72B0')
    bars2 = ax.bar(x + w/2, accs, w, label='Acc at Best F1 (%)', color='#55A868')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Best Validation Metrics — All Run4 Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
    ax.legend(); ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'best_metrics_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def plot_unbc_per_class(data):
    """Per-class F1 for UNBC experiments at best epoch."""
    unbc_keys = [k for k in UNBC_EXPS if k in data and data[k].get('per_class')]
    if not unbc_keys:
        return

    classes = ['No Pain', 'Mild Pain', 'Pain']
    x = np.arange(len(classes))
    n = len(unbc_keys)
    w = 0.8 / max(n, 1)
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, key in enumerate(unbc_keys):
        ep = data[key]['epochs']
        pc = data[key]['per_class']
        best_idx = max(range(len(ep)), key=lambda j: ep[j]['val_f1'])
        entry = pc[best_idx] if best_idx < len(pc) else pc[-1]
        vals = [entry['no_pain'], entry['mild_pain'], entry['pain']]
        bars = ax.bar(x + i * w - (n-1)*w/2, vals, w, label=data[key]['name'], color=cmap(i))
        for bar in bars:
            ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)

    ax.set_title('UNBC Per-Class F1 at Best Epoch (3-Class)', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 (%)'); ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.legend(fontsize=7); ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'unbc_per_class_f1.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def plot_demographics(data):
    """Plot demographic fairness from no-delta Swin experiment."""
    nodelta_path = SYNPAIN_EXPS.get('nodelta_swin', (None, None))[1]
    if not nodelta_path:
        return

    json_files = sorted(glob.glob(os.path.join(nodelta_path, 'epoch*_demographics.json')),
                        key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)))
    if not json_files:
        print('No demographic JSON files found for no-delta experiment, skipping.')
        return

    # Also load delta demographics for comparison
    delta_path = os.path.join(RESULTS_DIR, 'run1', 'synpain_delta_demo', 'bs_32_seed_0_lr_1e-05')
    delta_json_files = sorted(glob.glob(os.path.join(delta_path, 'epoch*_demographics.json')),
                              key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Demographic Fairness — No-Delta vs Delta', fontsize=16, fontweight='bold')

    def load_demo_data(json_files):
        epochs_data = []
        for jf in json_files:
            epoch_num = int(re.search(r'epoch(\d+)', jf).group(1))
            with open(jf) as f:
                d = json.load(f)
            d['epoch'] = epoch_num
            epochs_data.append(d)
        return epochs_data

    nodelta_data = load_demo_data(json_files)
    nd_epochs = [d['epoch'] for d in nodelta_data]

    # Age group F1 — no-delta
    ax = axes[0, 0]
    ax.plot(nd_epochs, [d['age_group']['Old']['f1']*100 for d in nodelta_data], marker='.', label='Old (no-delta)')
    ax.plot(nd_epochs, [d['age_group']['Young']['f1']*100 for d in nodelta_data], marker='.', label='Young (no-delta)')
    if delta_json_files:
        delta_data = load_demo_data(delta_json_files)
        d_epochs = [d['epoch'] for d in delta_data]
        ax.plot(d_epochs, [d['age_group']['Old']['f1']*100 for d in delta_data], marker='.', linestyle='--', label='Old (delta)', alpha=0.5)
        ax.plot(d_epochs, [d['age_group']['Young']['f1']*100 for d in delta_data], marker='.', linestyle='--', label='Young (delta)', alpha=0.5)
    ax.set_title('F1 by Age Group'); ax.set_xlabel('Epoch'); ax.set_ylabel('F1 (%)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Gender F1 — no-delta
    ax = axes[0, 1]
    ax.plot(nd_epochs, [d['gender']['F']['f1']*100 for d in nodelta_data], marker='.', label='Female (no-delta)')
    ax.plot(nd_epochs, [d['gender']['M']['f1']*100 for d in nodelta_data], marker='.', label='Male (no-delta)')
    if delta_json_files:
        ax.plot(d_epochs, [d['gender']['F']['f1']*100 for d in delta_data], marker='.', linestyle='--', label='Female (delta)', alpha=0.5)
        ax.plot(d_epochs, [d['gender']['M']['f1']*100 for d in delta_data], marker='.', linestyle='--', label='Male (delta)', alpha=0.5)
    ax.set_title('F1 by Gender'); ax.set_xlabel('Epoch'); ax.set_ylabel('F1 (%)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Bias gaps — no-delta
    ax = axes[1, 0]
    ax.plot(nd_epochs, [d['age_group_bias_gap_f1']*100 for d in nodelta_data], marker='.', label='Age Gap (no-delta)')
    ax.plot(nd_epochs, [d['gender_bias_gap_f1']*100 for d in nodelta_data], marker='.', label='Gender Gap (no-delta)')
    if delta_json_files:
        ax.plot(d_epochs, [d['age_group_bias_gap_f1']*100 for d in delta_data], marker='.', linestyle='--', label='Age Gap (delta)', alpha=0.5)
        ax.plot(d_epochs, [d['gender_bias_gap_f1']*100 for d in delta_data], marker='.', linestyle='--', label='Gender Gap (delta)', alpha=0.5)
    ax.set_title('F1 Bias Gap (pp)'); ax.set_xlabel('Epoch'); ax.set_ylabel('Gap (pp)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # AUROC by subgroup — no-delta best epoch
    ax = axes[1, 1]
    best = nodelta_data[-1]  # last epoch
    groups = ['Old', 'Young', 'Female', 'Male']
    aurocs = [best['age_group']['Old']['auroc']*100, best['age_group']['Young']['auroc']*100,
              best['gender']['F']['auroc']*100, best['gender']['M']['auroc']*100]
    bars = ax.bar(groups, aurocs, color=['#4C72B0', '#4C72B0', '#DD8452', '#DD8452'])
    for bar in bars:
        ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    ax.set_title(f'No-Delta AUROC by Subgroup (Epoch {best["epoch"]})'); ax.set_ylabel('AUROC (%)'); ax.set_ylim(80, 100); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'demographic_fairness.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


# ── Markdown ────────────────────────────────────────────────────────────────

def generate_markdown(data):
    md = []
    md.append('# Run 4: No-Delta & Part-Based Experiments\n')
    md.append('## Overview\n')
    md.append('This run investigates three questions:')
    md.append('1. **Does node-level delta matter?** Compare delta vs no-delta (graph-level) SynPAIN pretraining')
    md.append('2. **Does data composition matter?** Train on Part1-only (mostly pain) vs Part2-only (mostly no-pain)')
    md.append('3. **Demographic fairness without delta?** Compare bias gaps with vs without delta graph\n')

    # SynPAIN results
    md.append('## SynPAIN Pretraining Results\n')
    md.append('| Experiment | Best F1 (%) | Best Acc (%) | Best Epoch | Delta? |')
    md.append('|---|---|---|---|---|')
    for key in SYNPAIN_EXPS:
        if key not in data or not data[key]['epochs']:
            continue
        best = max(data[key]['epochs'], key=lambda e: e['val_f1'])
        is_delta = 'Yes' if 'delta' in key and 'nodelta' not in key else 'No'
        if 'part' in key:
            is_delta = 'Yes'
        md.append(f'| {data[key]["name"]} | **{best["val_f1"]:.2f}** | {best["val_acc"]:.2f} | {best["epoch"]} | {is_delta} |')

    md.append('')
    md.append('**Key findings:**')
    for key in SYNPAIN_EXPS:
        if key in data and data[key]['epochs']:
            best = max(data[key]['epochs'], key=lambda e: e['val_f1'])
            md.append(f'- {data[key]["name"]}: best F1 = **{best["val_f1"]:.2f}%** (epoch {best["epoch"]})')
    md.append('')

    # UNBC results
    unbc_keys = [k for k in UNBC_EXPS if k in data and data[k]['epochs']]
    if unbc_keys:
        md.append('## UNBC 3-Class Finetuning Results\n')
        md.append('| Experiment | Best F1 (%) | Best Acc (%) | Best Epoch |')
        md.append('|---|---|---|---|')
        for key in unbc_keys:
            best = max(data[key]['epochs'], key=lambda e: e['val_f1'])
            md.append(f'| {data[key]["name"]} | **{best["val_f1"]:.2f}** | {best["val_acc"]:.2f} | {best["epoch"]} |')

        md.append('')
        md.append('### Per-Class F1 at Best Epoch\n')
        md.append('| Experiment | No Pain F1 (%) | Mild Pain F1 (%) | Pain F1 (%) |')
        md.append('|---|---|---|---|')
        for key in unbc_keys:
            if not data[key].get('per_class'):
                continue
            ep = data[key]['epochs']
            best_idx = max(range(len(ep)), key=lambda j: ep[j]['val_f1'])
            pc = data[key]['per_class']
            entry = pc[best_idx] if best_idx < len(pc) else pc[-1]
            md.append(f'| {data[key]["name"]} | {entry["no_pain"]:.1f} | {entry["mild_pain"]:.1f} | {entry["pain"]:.1f} |')

        md.append('')
        md.append('**Key findings:**')
        for key in unbc_keys:
            best = max(data[key]['epochs'], key=lambda e: e['val_f1'])
            md.append(f'- {data[key]["name"]}: best F1 = **{best["val_f1"]:.2f}%** (epoch {best["epoch"]})')
        md.append('')

    # Demographics
    nodelta_path = SYNPAIN_EXPS.get('nodelta_swin', (None, None))[1]
    if nodelta_path:
        json_files = sorted(glob.glob(os.path.join(nodelta_path, 'epoch*_demographics.json')),
                            key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)))
        if json_files:
            best_json = json_files[-1]
            best_epoch = int(re.search(r'epoch(\d+)', best_json).group(1))
            with open(best_json) as f:
                d = json.load(f)
            md.append('## Demographic Fairness — No-Delta Swin-B\n')
            md.append(f'### Epoch {best_epoch} Subgroup Performance\n')
            md.append('| Group | Subgroup | N | AUROC | F1 | Balanced Acc |')
            md.append('|---|---|---|---|---|---|')
            for group in ['age_group', 'gender']:
                for subgroup, metrics in d[group].items():
                    label = 'Female' if subgroup == 'F' else ('Male' if subgroup == 'M' else subgroup)
                    md.append(f'| {group.replace("_", " ").title()} | {label} | {metrics["n_samples"]} | {metrics["auroc"]*100:.1f} | {metrics["f1"]*100:.1f} | {metrics["balanced_acc"]*100:.1f} |')

            md.append('')
            md.append('### Bias Gaps\n')
            md.append('| Dimension | F1 Gap (pp) | AUROC Gap (pp) |')
            md.append('|---|---|---|')
            md.append(f'| Age Group | {d["age_group_bias_gap_f1"]*100:.2f} | {d["age_group_bias_gap_auroc"]*100:.2f} |')
            md.append(f'| Gender | {d["gender_bias_gap_f1"]*100:.2f} | {d["gender_bias_gap_auroc"]*100:.2f} |')
            md.append('')

    # Plots
    md.append('## Plots\n')
    md.append('### SynPAIN Delta vs No-Delta')
    md.append('![SynPAIN Comparison](synpain_comparison.png)\n')
    md.append('### UNBC Finetuning Comparison')
    md.append('![UNBC Comparison](unbc_comparison.png)\n')
    md.append('### Best Metrics')
    md.append('![Best Metrics](best_metrics_comparison.png)\n')
    md.append('### UNBC Per-Class F1')
    md.append('![Per-Class F1](unbc_per_class_f1.png)\n')
    md.append('### Demographic Fairness')
    md.append('![Demographics](demographic_fairness.png)\n')

    path = os.path.join(OUT_DIR, 'summary.md')
    with open(path, 'w') as f:
        f.write('\n'.join(md))
    print(f'Saved {path}')


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading experiments...')
    data = load_all()
    print(f'Loaded {len(data)} experiments\n')

    plot_synpain_comparison(data)
    plot_unbc_comparison(data)
    plot_best_metrics_bar(data)
    plot_unbc_per_class(data)
    plot_demographics(data)
    generate_markdown(data)
    print('\nDone!')
