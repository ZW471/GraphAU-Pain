import os
from tqdm import tqdm

unbc_label = [4, 6, 7, 9, 10, 12, 20, 25, 26, 43]
disfa_label = [1, 2, 4, 6, 9, 12, 25, 26]
mapping = [None, None, 0, 1, 3, 5, 7, 8]


with open('../data/UNBC/disfa_labelled/unbc_predictions.txt', 'r') as pred_file:
    pred_labels = pred_file.readlines()

with open('../data/UNBC/disfa_labelled/UNBC_test_label_fold1.txt', 'r') as ori_file:
    ori_labels = ori_file.readlines()


with open('../data/UNBC/disfa_labelled/UNBC_disfa_labelled.txt', 'w') as new_file:
    new_file.write('')
with open('../data/UNBC/disfa_labelled/UNBC_disfa_labelled.txt', 'a') as new_file:
    for ori, pred in tqdm(zip(ori_labels, pred_labels)):
        ori = ori.strip().split(' ')
        pred = pred.strip().split(' ')
        new_label = []
        for idx, label in enumerate(pred):
            if mapping[idx] is not None:
                new_label.append(str(int(ori[mapping[idx]])))
            else:
                new_label.append(str(ori[idx]))
        new_file.write(' '.join(new_label) + '\n')