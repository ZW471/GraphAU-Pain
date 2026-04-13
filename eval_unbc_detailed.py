"""Evaluate UNBC pain estimation checkpoint with F1, Acc, Precision, Recall.

Usage:
    python eval_unbc_detailed.py <ckpt_path> <arc> <crop_size> [generic|orig|headless] [fold] [label_path] [num_classes]

Outputs JSON to stdout: per-class P/R/F1/Acc plus mean.
"""
import json
import sys
import torch
from torch.utils.data import DataLoader

from dataset import UNBC
from utils import image_test, statistics_softmax, update_statistics_list, load_state_dict
from model.ANFL import FullPictureMEFARG, FullPictureMEFARGGeneric, BackboneOnlyPain


def main():
    ckpt = sys.argv[1]
    arc = sys.argv[2]
    crop_size = int(sys.argv[3])
    model_kind = sys.argv[4] if len(sys.argv) > 4 else 'generic'
    fold = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    label_path = sys.argv[6] if len(sys.argv) > 6 else ''
    num_classes = int(sys.argv[7]) if len(sys.argv) > 7 else 8

    valset = UNBC('data/UNBC', train=False, fold=fold,
                  transform=image_test(crop_size=crop_size), stage=3, label_path=label_path)
    val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)

    if model_kind == 'orig':
        net = FullPictureMEFARG(num_classes=num_classes, backbone=arc, neighbor_num=4, metric='dots', binary=False)
    elif model_kind == 'headless':
        net = BackboneOnlyPain(num_classes=num_classes, backbone=arc, neighbor_num=4, metric='dots', binary=False)
    else:
        net = FullPictureMEFARGGeneric(num_classes=num_classes, backbone=arc, neighbor_num=4, metric='dots', binary=False)
    net = load_state_dict(net, ckpt)
    net = net.cuda().eval()

    stats = None
    with torch.no_grad():
        for inputs, targets in val_loader:
            targets = targets.float().cuda()
            inputs = inputs.cuda()
            outputs = net(inputs)
            update = statistics_softmax(outputs, targets.detach())
            stats = update_statistics_list(stats, update)

    out = {'per_class': [], 'mean': {}}
    p_list, r_list, f1_list, acc_list = [], [], [], []
    for i, s in enumerate(stats):
        TP, FP, FN, TN = s['TP'], s['FP'], s['FN'], s['TN']
        p = TP / (TP + FP + 1e-20)
        r = TP / (TP + FN + 1e-20)
        f1 = 2 * p * r / (p + r + 1e-20)
        acc = (TP + TN) / (TP + TN + FP + FN)
        out['per_class'].append({
            'class': i, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'precision': p, 'recall': r, 'f1': f1, 'acc': acc,
        })
        p_list.append(p); r_list.append(r); f1_list.append(f1); acc_list.append(acc)
    out['mean'] = {
        'precision': sum(p_list) / len(p_list),
        'recall': sum(r_list) / len(r_list),
        'f1': sum(f1_list) / len(f1_list),
        'acc': sum(acc_list) / len(acc_list),
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
