"""Run UNBC pain detailed eval (P/R/F1/Acc) for a list of experiments.

Reads a registry of (name, log_path, ckpt_dir, arc, crop_size, model_kind, [label_path], [num_classes]),
parses the log to find the best epoch by val_mean_f1_score, loads the
matching epoch{N}_model_fold1.pth checkpoint, runs validation, and writes
a per-experiment JSON to summary/run5-comprehensive/eval/<name>.json.

label_path (optional 7th tuple element) is forwarded to eval_unbc_detailed.py
so the eval pulls from data/UNBC/list/<label_path>/ — required for run6/run7
which use original_unbc labels.

num_classes (optional 8th tuple element, default 8) sets the AU GNN node
count for model construction — required for run7 baselines which use 10
nodes to match the original UNBC AU vocabulary.
"""
import json
import os
import re
import subprocess
import sys


REGISTRY = [
    # name, log, ckpt_dir, arc, crop, model_kind
    # ---- R50 (FullPictureMEFARG, crop=172) ----
    ("unbc_baseline_r50",
     "results/run-3class/unbc_finetune_3class_fold1_run.log",
     "results/run-3class/unbc_finetune_3class_fold1/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig"),
    ("unbc_from_disfa_fold3_r50",
     "results/run-3class/unbc_3class_from_disfa_f3_fold1_run.log",
     "results/run-3class/unbc_3class_from_disfa_f3_fold1/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig"),
    ("unbc_from_disfa_full_r50",
     "results/run5/unbc_from_disfa_full_r50_run.log",
     "results/run5/unbc_from_disfa_full_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig"),
    ("unbc_from_delta_full_r50",
     "results/run-3class/unbc_3class_from_synpain_r50_fold1_run.log",
     "results/run-3class/unbc_3class_from_synpain_r50_fold1/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig"),
    ("unbc_from_nodelta_full_r50",
     "results/unbc_3class_from_nodelta_r50_fold1_run.log",
     "results/unbc_3class_from_nodelta_r50_fold1/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig"),
    ("unbc_from_part1_r50",
     "results/unbc_3class_from_part1_r50_fold1_run.log",
     "results/unbc_3class_from_part1_r50_fold1/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig"),
    ("unbc_from_part2_r50",
     "results/unbc_3class_from_part2_r50_fold1_run.log",
     "results/unbc_3class_from_part2_r50_fold1/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig"),
    # ---- Swin (FullPictureMEFARGGeneric, crop=224) ----
    ("unbc_baseline_swin",
     "results/run5/unbc_baseline_swin_run.log",
     "results/run5/unbc_baseline_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic"),
    ("unbc_from_disfa_full_swin",
     "results/run5/unbc_from_disfa_full_swin_run.log",
     "results/run5/unbc_from_disfa_full_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic"),
    ("unbc_from_delta_full_swin",
     "results/run5/unbc_from_delta_full_swin_run.log",
     "results/run5/unbc_from_delta_full_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic"),
    ("unbc_from_nodelta_full_swin",
     "results/run5/unbc_from_nodelta_swin_run.log",
     "results/run5/unbc_from_nodelta_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic"),
    ("unbc_from_part1_swin",
     "results/run5/unbc_from_part1_swin_run.log",
     "results/run5/unbc_from_part1_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic"),
    ("unbc_from_part2_swin",
     "results/run5/unbc_from_part2_swin_run.log",
     "results/run5/unbc_from_part2_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic"),
    # ---- Round-5 NoΔ Part1/Part2 → UNBC ----
    ("unbc_from_nodelta_part1_r50",
     "results/run5/unbc_from_nodelta_part1_r50_run.log",
     "results/run5/unbc_from_nodelta_part1_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig"),
    ("unbc_from_nodelta_part2_r50",
     "results/run5/unbc_from_nodelta_part2_r50_run.log",
     "results/run5/unbc_from_nodelta_part2_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig"),
    ("unbc_from_nodelta_part1_swin",
     "results/run5/unbc_from_nodelta_part1_swin_run.log",
     "results/run5/unbc_from_nodelta_part1_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic"),
    ("unbc_from_nodelta_part2_swin",
     "results/run5/unbc_from_nodelta_part2_swin_run.log",
     "results/run5/unbc_from_nodelta_part2_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic"),
    # ---- Round-4 R50 improvement attempts ----
    ("unbc_from_delta_r50_lowlr",
     "results/run5/unbc_from_delta_r50_lowlr_run.log",
     "results/run5/unbc_from_delta_r50_lowlr/bs_64_seed_0_lr_3e-05",
     "resnet50", 172, "orig"),
    ("unbc_from_delta_r50_smallbs",
     "results/run5/unbc_from_delta_r50_smallbs_run.log",
     "results/run5/unbc_from_delta_r50_smallbs/bs_32_seed_0_lr_5e-05",
     "resnet50", 172, "orig"),
    ("unbc_from_part2_r50_lowlr",
     "results/run5/unbc_from_part2_r50_lowlr_run.log",
     "results/run5/unbc_from_part2_r50_lowlr/bs_64_seed_0_lr_3e-05",
     "resnet50", 172, "orig"),
    # ---- Run 6: original_unbc labels (--label_path original_unbc) ----
    ("run6_unbc_baseline_r50",
     "results/run6/unbc_baseline_r50_run.log",
     "results/run6/unbc_baseline_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig", "original_unbc"),
    ("run6_unbc_baseline_swin",
     "results/run6/unbc_baseline_swin_run.log",
     "results/run6/unbc_baseline_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc"),
    ("run6_unbc_from_nodelta_full_r50",
     "results/run6/unbc_from_nodelta_full_r50_run.log",
     "results/run6/unbc_from_nodelta_full_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig", "original_unbc"),
    ("run6_unbc_from_nodelta_full_swin",
     "results/run6/unbc_from_nodelta_full_swin_run.log",
     "results/run6/unbc_from_nodelta_full_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc"),
    ("run6_unbc_from_nodelta_part1_r50",
     "results/run6/unbc_from_nodelta_part1_r50_run.log",
     "results/run6/unbc_from_nodelta_part1_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig", "original_unbc"),
    ("run6_unbc_from_nodelta_part1_swin",
     "results/run6/unbc_from_nodelta_part1_swin_run.log",
     "results/run6/unbc_from_nodelta_part1_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc"),
    ("run6_unbc_from_nodelta_part2_r50",
     "results/run6/unbc_from_nodelta_part2_r50_run.log",
     "results/run6/unbc_from_nodelta_part2_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig", "original_unbc"),
    ("run6_unbc_from_nodelta_part2_swin",
     "results/run6/unbc_from_nodelta_part2_swin_run.log",
     "results/run6/unbc_from_nodelta_part2_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc"),
    # ---- Run 7: 10-node baselines + head-less SynPAIN→UNBC ----
    # Baselines use 10-node AU graph (matches original UNBC AU vocab) with
    # the standard FullPictureMEFARG{,Generic} head.
    ("run7_unbc_baseline_r50",
     "results/run7/unbc_baseline_r50_run.log",
     "results/run7/unbc_baseline_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig", "original_unbc", 10),
    ("run7_unbc_baseline_swin",
     "results/run7/unbc_baseline_swin_run.log",
     "results/run7/unbc_baseline_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    # SynPAIN→UNBC: head-less (BackboneOnlyPain). num_classes is irrelevant
    # for this model class but we pass 10 for consistency with the baselines.
    ("run7_unbc_from_synpain_full_r50",
     "results/run7/unbc_from_synpain_full_r50_run.log",
     "results/run7/unbc_from_synpain_full_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "headless", "original_unbc", 10),
    ("run7_unbc_from_synpain_full_swin",
     "results/run7/unbc_from_synpain_full_swin_run.log",
     "results/run7/unbc_from_synpain_full_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "headless", "original_unbc", 10),
    ("run7_unbc_from_synpain_part1_r50",
     "results/run7/unbc_from_synpain_part1_r50_run.log",
     "results/run7/unbc_from_synpain_part1_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "headless", "original_unbc", 10),
    ("run7_unbc_from_synpain_part1_swin",
     "results/run7/unbc_from_synpain_part1_swin_run.log",
     "results/run7/unbc_from_synpain_part1_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "headless", "original_unbc", 10),
    ("run7_unbc_from_synpain_part2_r50",
     "results/run7/unbc_from_synpain_part2_r50_run.log",
     "results/run7/unbc_from_synpain_part2_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "headless", "original_unbc", 10),
    ("run7_unbc_from_synpain_part2_swin",
     "results/run7/unbc_from_synpain_part2_swin_run.log",
     "results/run7/unbc_from_synpain_part2_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "headless", "original_unbc", 10),
    # ---- Run 8: SynPAIN→UNBC with the same FullPictureMEFARG{,Generic} model
    # as the UNBC baselines (no head-less). The SynPAIN pretrain stage uses
    # pain-only supervision (no AU loss; the model has no AU classifier, just
    # an AU graph featurizer wired into the pain logit). 10-node head, nb=4.
    # NOTE: run8 chain jobs use a single shared stdout log per chain, but each
    # phase still writes its own train.log inside its outdir, so we point at
    # `<outdir>/train.log` to isolate finetune-only epochs.
    ("run8_unbc_from_synpain_pain_full_r50",
     "results/run8/unbc_from_synpain_pain_full_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run8/unbc_from_synpain_pain_full_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig", "original_unbc", 10),
    ("run8_unbc_from_synpain_pain_full_swin",
     "results/run8/unbc_from_synpain_pain_full_swin/bs_32_seed_0_lr_0.0001/train.log",
     "results/run8/unbc_from_synpain_pain_full_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    ("run8_unbc_from_synpain_pain_part1_r50",
     "results/run8/unbc_from_synpain_pain_part1_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run8/unbc_from_synpain_pain_part1_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig", "original_unbc", 10),
    ("run8_unbc_from_synpain_pain_part1_swin",
     "results/run8/unbc_from_synpain_pain_part1_swin/bs_32_seed_0_lr_0.0001/train.log",
     "results/run8/unbc_from_synpain_pain_part1_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    ("run8_unbc_from_synpain_pain_part2_r50",
     "results/run8/unbc_from_synpain_pain_part2_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run8/unbc_from_synpain_pain_part2_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig", "original_unbc", 10),
    ("run8_unbc_from_synpain_pain_part2_swin",
     "results/run8/unbc_from_synpain_pain_part2_swin/bs_32_seed_0_lr_0.0001/train.log",
     "results/run8/unbc_from_synpain_pain_part2_swin/bs_32_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    # ---- Run 8: DISFA→UNBC R50 investigation (no code mods, varying labels/
    # head-shape/lr/forward to chase the expected ~66% F1).
    ("run8_disfa_to_unbc_r50_ori10",
     "results/run8/disfa_to_unbc_r50_ori10/bs_64_seed_0_lr_0.0001/train.log",
     "results/run8/disfa_to_unbc_r50_ori10/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig", "original_unbc", 10),
    ("run8_disfa_to_unbc_r50_ori8",
     "results/run8/disfa_to_unbc_r50_ori8/bs_64_seed_0_lr_0.0001/train.log",
     "results/run8/disfa_to_unbc_r50_ori8/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "orig", "original_unbc", 8),
    ("run8_disfa_to_unbc_r50_ori10_lr1e5",
     "results/run8/disfa_to_unbc_r50_ori10_lr1e5/bs_64_seed_0_lr_1e-05/train.log",
     "results/run8/disfa_to_unbc_r50_ori10_lr1e5/bs_64_seed_0_lr_1e-05",
     "resnet50", 172, "orig", "original_unbc", 10),
    ("run8_disfa_to_unbc_r50_ori10_generic",
     "results/run8/disfa_to_unbc_r50_ori10_generic/bs_64_seed_0_lr_0.0001/train.log",
     "results/run8/disfa_to_unbc_r50_ori10_generic/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    # ---- Run 9: backbone-only SynPAIN pretrain → UNBC finetune with AU aux loss
    # Uses FullPictureMEFARGAuAux (identical state_dict to FullPictureMEFARGGeneric).
    # Eval loads into Generic (kind='generic'). 10-node head, original_unbc labels.
    ("run9_unbc_au_aux_from_full_r50",
     "results/run9/unbc_au_aux_from_full_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_au_aux_from_full_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    ("run9_unbc_au_aux_from_full_swin",
     "results/run9/unbc_au_aux_from_full_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_au_aux_from_full_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    ("run9_unbc_au_aux_from_part1_r50",
     "results/run9/unbc_au_aux_from_part1_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_au_aux_from_part1_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    ("run9_unbc_au_aux_from_part1_swin",
     "results/run9/unbc_au_aux_from_part1_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_au_aux_from_part1_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    ("run9_unbc_au_aux_from_part2_r50",
     "results/run9/unbc_au_aux_from_part2_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_au_aux_from_part2_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    ("run9_unbc_au_aux_from_part2_swin",
     "results/run9/unbc_au_aux_from_part2_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_au_aux_from_part2_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    # Run 9: UNBC with DISFA-derived labels (8-AU, default list/) + AU aux loss
    ("run9_unbc_au_aux_disfa_labels_r50",
     "results/run9/unbc_au_aux_disfa_labels_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_au_aux_disfa_labels_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "", 8),
    ("run9_unbc_au_aux_disfa_labels_swin",
     "results/run9/unbc_au_aux_disfa_labels_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_au_aux_disfa_labels_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "", 8),
    # Run 9: no-AU-aux baselines (lam=0) with FullPictureMEFARGAuAux
    ("run9_unbc_no_aux_disfa_labels_r50",
     "results/run9/unbc_no_aux_disfa_labels_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_no_aux_disfa_labels_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "", 8),
    ("run9_unbc_no_aux_disfa_labels_swin",
     "results/run9/unbc_no_aux_disfa_labels_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_no_aux_disfa_labels_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "", 8),
    ("run9_unbc_no_aux_ori_labels_r50",
     "results/run9/unbc_no_aux_ori_labels_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_no_aux_ori_labels_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    ("run9_unbc_no_aux_ori_labels_swin",
     "results/run9/unbc_no_aux_ori_labels_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_no_aux_ori_labels_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    # Run 9 baselines: UNBC-only with AU aux loss (no pretrain)
    ("run9_unbc_au_aux_only_r50",
     "results/run9/unbc_au_aux_only_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_au_aux_only_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    ("run9_unbc_au_aux_only_swin",
     "results/run9/unbc_au_aux_only_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run9/unbc_au_aux_only_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    # ---- run10: SynPAIN pretrain (composite-split fix) → UNBC AU aux ----
    # Headless pretrain → UNBC
    ("run10_unbc_from_hl_full_r50",
     "results/run10/unbc_from_hl_full_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_hl_full_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    ("run10_unbc_from_hl_full_swin",
     "results/run10/unbc_from_hl_full_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_hl_full_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    ("run10_unbc_from_hl_part1_r50",
     "results/run10/unbc_from_hl_part1_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_hl_part1_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    ("run10_unbc_from_hl_part1_swin",
     "results/run10/unbc_from_hl_part1_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_hl_part1_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    ("run10_unbc_from_hl_part2_r50",
     "results/run10/unbc_from_hl_part2_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_hl_part2_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    ("run10_unbc_from_hl_part2_swin",
     "results/run10/unbc_from_hl_part2_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_hl_part2_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    # Full-model pretrain → UNBC
    ("run10_unbc_from_fm_full_r50",
     "results/run10/unbc_from_fm_full_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_fm_full_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    ("run10_unbc_from_fm_full_swin",
     "results/run10/unbc_from_fm_full_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_fm_full_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    ("run10_unbc_from_fm_part1_r50",
     "results/run10/unbc_from_fm_part1_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_fm_part1_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    ("run10_unbc_from_fm_part1_swin",
     "results/run10/unbc_from_fm_part1_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_fm_part1_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
    ("run10_unbc_from_fm_part2_r50",
     "results/run10/unbc_from_fm_part2_r50/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_fm_part2_r50/bs_64_seed_0_lr_0.0001",
     "resnet50", 172, "generic", "original_unbc", 10),
    ("run10_unbc_from_fm_part2_swin",
     "results/run10/unbc_from_fm_part2_swin/bs_64_seed_0_lr_0.0001/train.log",
     "results/run10/unbc_from_fm_part2_swin/bs_64_seed_0_lr_0.0001",
     "swin_transformer_base", 224, "generic", "original_unbc", 10),
]


EPOCH_RE = re.compile(r"'Epoch:\s+(\d+)\s+train_loss:.*?val_mean_f1_score\s+([\d.]+),val_mean_acc\s+([\d.]+)'")


def best_epoch(log_path):
    """Return (best_epoch, best_f1, best_acc) by max val_mean_f1_score."""
    if not os.path.exists(log_path):
        return None
    best = None
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                ep, f1, acc = int(m.group(1)), float(m.group(2)), float(m.group(3))
                if best is None or f1 > best[1]:
                    best = (ep, f1, acc)
    return best


def main():
    out_dir = "summary/run5-comprehensive/eval"
    os.makedirs(out_dir, exist_ok=True)
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    # Load existing summary so single-name runs MERGE rather than overwrite.
    summary_path = os.path.join(out_dir, "_summary.json")
    existing_by_name = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                for r in json.load(f):
                    existing_by_name[r['name']] = r
        except Exception:
            pass
    summary_rows = []

    for entry in REGISTRY:
        name, log, ckpt_dir, arc, crop, kind = entry[:6]
        label_path = entry[6] if len(entry) > 6 else ''
        num_classes = entry[7] if len(entry) > 7 else 8
        if only is not None and name not in only:
            continue
        be = best_epoch(log)
        if be is None:
            print(f"[SKIP {name}] no log {log}")
            continue
        ep, f1, acc = be
        ckpt = os.path.join(ckpt_dir, f"epoch{ep}_model_fold1.pth")
        if not os.path.exists(ckpt):
            print(f"[SKIP {name}] no ckpt {ckpt}")
            continue
        out_json = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(out_json):
            print(f"[CACHED {name}] {out_json}")
        else:
            print(f"[EVAL {name}] epoch={ep} log_f1={f1:.2f} log_acc={acc:.2f} "
                  f"label_path={label_path or '(default)'} nc={num_classes}")
            cmd = ["uv", "run", "python", "eval_unbc_detailed.py",
                   ckpt, arc, str(crop), kind, "1", label_path, str(num_classes)]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                with open(out_json, 'w') as f:
                    f.write(proc.stdout)
            except subprocess.CalledProcessError as e:
                print(f"[FAIL {name}] {e.stderr[-500:]}")
                continue

        with open(out_json) as f:
            data = json.load(f)
        m = data['mean']
        summary_rows.append({
            'name': name, 'arc': arc, 'best_epoch': ep,
            'log_f1': f1, 'log_acc': acc,
            'eval_f1': m['f1'] * 100, 'eval_acc': m['acc'] * 100,
            'eval_precision': m['precision'] * 100, 'eval_recall': m['recall'] * 100,
            'per_class_f1': [c['f1'] * 100 for c in data['per_class']],
            'per_class_p': [c['precision'] * 100 for c in data['per_class']],
            'per_class_r': [c['recall'] * 100 for c in data['per_class']],
            'per_class_acc': [c['acc'] * 100 for c in data['per_class']],
        })

    # Merge new rows into existing, preserving order from REGISTRY.
    by_name = dict(existing_by_name)
    for r in summary_rows:
        by_name[r['name']] = r
    ordered = []
    seen = set()
    for entry in REGISTRY:
        n = entry[0]
        if n in by_name and n not in seen:
            row = by_name[n]
            # Stamp label_path from REGISTRY so the eval row records which split it was scored on.
            if len(entry) > 6 and entry[6]:
                row['label_path'] = entry[6]
            ordered.append(row); seen.add(n)
    # Append any leftover names not in REGISTRY.
    for n, r in by_name.items():
        if n not in seen:
            ordered.append(r); seen.add(n)
    with open(summary_path, 'w') as f:
        json.dump(ordered, f, indent=2)
    print(f"\nMerged {len(summary_rows)} new rows; total {len(ordered)} in {summary_path}")


if __name__ == '__main__':
    main()
