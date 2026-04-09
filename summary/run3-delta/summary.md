# GraphAU-Pain Extension - Delta Graph Experiment Summary

## Overview

This document summarizes the results of testing `pain_estimation_delta.py` (DeltaMEFARG) on the UNBC dataset for 3-class pain classification (No Pain / Mild Pain / Pain).

**Experiment design:** 2x4 matrix comparing:
- **Delta mode:** Node-level ΔGraph (`--use_delta_graph`) vs PE-score delta (graph-level)
- **Pretraining:** DISFA AU fold 1/2/3 vs no pretraining (from scratch)

**Setup:** ResNet-50 backbone, fold 1, batch size 64, LR 1e-4, crop size 172, 20 epochs, AdamW optimizer with cosine LR decay.

---

## Delta Results (3-Class: No Pain / Mild Pain / Pain)

### Node-Level ΔGraph (`--use_delta_graph`)

| Experiment | Best Mean F1 (%) | Best Acc (%) | Best Epoch |
|---|---|---|---|
| ΔGraph + DISFA f1 | 31.13 | 63.63 | 11 |
| ΔGraph + DISFA f2 | **32.05** | 65.51 | 2 |
| ΔGraph + DISFA f3 | 31.01 | 60.75 | 1 |
| ΔGraph (no pretrain) | 29.06 | 58.94 | 2 |

### Graph-Level PE-Score Delta (no `--use_delta_graph`)

| Experiment | Best Mean F1 (%) | Best Acc (%) | Best Epoch |
|---|---|---|---|
| PE-score + DISFA f1 | 27.47 | 58.95 | 4 |
| PE-score + DISFA f2 | 29.72 | 62.24 | 4 |
| PE-score + DISFA f3 | **34.66** | 67.51 | 9 |
| PE-score (no pretrain) | 34.23 | 65.68 | 6 |

### Per-Class F1 at Best Epoch

| Experiment | No Pain F1 (%) | Mild Pain F1 (%) | Pain F1 (%) | Mean F1 (%) |
|---|---|---|---|---|
| ΔGraph + DISFA f1 | 61.5 | 26.6 | 5.3 | 31.13 |
| ΔGraph + DISFA f2 | 64.8 | 23.6 | 7.7 | 32.05 |
| ΔGraph + DISFA f3 | 54.5 | 35.2 | 3.4 | 31.01 |
| ΔGraph (no pretrain) | 52.2 | 29.9 | 5.1 | 29.06 |
| PE-score + DISFA f1 | 54.1 | 23.5 | 4.9 | 27.47 |
| PE-score + DISFA f2 | 58.5 | 27.7 | 3.0 | 29.72 |
| PE-score + DISFA f3 | 67.7 | 31.3 | 5.0 | 34.66 |
| PE-score (no pretrain) | 66.2 | 31.8 | 4.8 | 34.23 |

---

## Comparison with FullPictureMEFARG Baselines (run2-3class)

| Method | Pretrain | Best Mean F1 (%) | Best Acc (%) | Best Epoch |
|---|---|---|---|---|
| **FullPictureMEFARG** (baseline) | None | 53.71 | 79.87 | 2 |
| **FullPictureMEFARG** | DISFA f1 AU | 55.51 | 87.71 | 7 |
| **FullPictureMEFARG** | DISFA f2 AU | 57.98 | 89.97 | 5 |
| **FullPictureMEFARG** | DISFA f3 AU | **62.74** | 88.41 | 4 |
| **FullPictureMEFARG** | UNBC AU | 54.50 | 88.78 | 3 |
| DeltaMEFARG ΔGraph | DISFA f2 AU | 32.05 | 65.51 | 2 |
| DeltaMEFARG PE-score | DISFA f3 AU | 34.66 | 67.51 | 9 |
| DeltaMEFARG PE-score | None | 34.23 | 65.68 | 6 |

---

## Key Findings

### 1. DeltaMEFARG substantially underperforms FullPictureMEFARG on UNBC

The best delta model (PE-score + DISFA f3, 34.66% F1) achieves only **55% of the best baseline** (FullPictureMEFARG + DISFA f3, 62.74% F1). This ~28 pp gap is large and consistent across all configurations.

### 2. The delta approach struggles with single-frame pseudo-pairing

UNBC provides single frames, not genuine neutral-expressive pairs. The script auto-constructs pseudo-pairs by selecting one neutral reference per subject. This creates two problems:
- The neutral reference may not be truly neutral (just the lowest-pain frame available)
- The per-subject neutral is fixed, so all frames from the same subject share the same delta baseline — reducing the model's ability to discriminate within-subject pain variation

### 3. PE-score delta outperforms node-level ΔGraph on UNBC

Surprisingly, the simpler graph-level PE-score delta (best: 34.66%) outperforms the node-level ΔGraph (best: 32.05%). This reverses the pattern seen on SynPAIN (where ΔGraph > PE-score). Possible explanation: the noisy pseudo-pairing introduces per-node noise that accumulates in the ΔGraph subtraction, while the pooled PE-score is more robust.

### 4. DISFA pretraining provides inconsistent benefit for delta models

- For ΔGraph: DISFA f2 (32.05%) slightly outperforms no pretrain (29.06%), but f1 and f3 do not consistently help
- For PE-score: DISFA f3 (34.66%) slightly outperforms no pretrain (34.23%), but f1 underperforms
- Overall: pretraining benefit is marginal (+0-3 pp) compared to the ~9 pp boost seen with FullPictureMEFARG

### 5. Pain class remains extremely difficult

All delta models achieve very low Pain F1 (3-8%), much worse than the baseline (31-51%). The delta representation fails to capture the pain signal, likely because the subtraction with a fixed neutral reference removes useful absolute-intensity information.

### 6. Training is unstable with high variance

F1 scores fluctuate substantially across epochs (e.g., PE-score+DISFA f3 ranges from 24.4% to 34.7%). Early stopping is critical, but the optimal epoch is hard to predict.

---

## Conclusion

The DeltaMEFARG approach, while effective for genuine paired data (SynPAIN), **does not improve pain estimation on UNBC** when using auto-constructed pseudo-pairs. The fixed-neutral pairing strategy does not provide the same quality of contrast as SynPAIN's true neutral-expressive pairs. For UNBC 3-class pain classification, FullPictureMEFARG with DISFA AU pretraining remains the best approach (62.74% mean F1).

**Recommendation:** The delta approach should only be used when genuine paired data (neutral + expressive from the same session) is available. For single-frame datasets like UNBC, the standard full-picture model is preferred.
