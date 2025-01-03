import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from model.MEFL import MEFARG
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env


def get_dataloader(conf):
    print('==> Preparing data...')

    if conf.dataset == 'UNBC':
        valset = UNBC(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return val_loader, len(valset)


# Val
def val(net, val_loader, output_prediction=None):
    if output_prediction is not None:
        with open(output_prediction, 'w') as f:
            f.write('')
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        targets = targets.float()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net(inputs)
            if output_prediction is not None:
                with open(output_prediction, 'a') as f:
                    output_texts = outputs.cpu().numpy()
                    output_texts *= 5
                    for i in range(output_texts.shape[0]):
                        f.write(' '.join([str(round(elem, 2)) for elem in output_texts[i]]) + '\n')
                with open(f'{output_prediction.split(".")[0]}_target.{output_prediction.split(".")[1]}', 'a') as f:
                    target_texts = targets.detach().cpu().numpy()
                    for i in range(target_texts.shape[0]):
                        f.write(' '.join([str(round(elem, 2)) for elem in target_texts[i]]) + '\n')
    # return mean_f1_score, f1_score_list, mean_acc, acc_list
    return [], [], [], []

def main(conf):
    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist
    elif conf.dataset == 'UNBC':
        dataset_info = UNBC_infolist

    # data
    val_loader, val_data_num = get_dataloader(conf)
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))
    net = MEFARG(num_classes=conf.num_classes, backbone=conf.arc)

    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    #test
    val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, output_prediction=conf.prediction)

    # log
    # infostr = {'val_mean_f1_score {:.2f} val_mean_acc {:.2f}' .format(100.* val_mean_f1_score, 100.* val_mean_acc)}
    # logging.info(infostr)
    # infostr = {'F1-score-list:'}
    # logging.info(infostr)
    # infostr = dataset_info(val_f1_score)
    # logging.info(infostr)
    # infostr = {'Acc-list:'}
    # logging.info(infostr)
    # infostr = dataset_info(val_acc)
    # logging.info(infostr)



# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

