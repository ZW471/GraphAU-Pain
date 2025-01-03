import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging

from model.ANFL import MEFARG, PainEstimation, BackboneOnly, FullPictureMEFARGNoGNN, FullPictureMEFARG, \
    RegFullPictureMEFARG
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env


def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'UNBC':
        trainset = UNBC(conf.dataset_path, train=True, fold = conf.fold, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 3)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        valset = UNBC(conf.dataset_path, train=False, fold=conf.fold, transform=image_test(crop_size=conf.crop_size), stage = 3)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return train_loader, val_loader, len(trainset), len(valset)


# Train
def train(conf,net,train_loader,optimizer,epoch,criterion):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs,  targets) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, train_loader_len)
        targets = targets.float()
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
    return losses.avg


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
            outputs = net(inputs)
            if output_prediction is not None:
                # TODO:
                with open(output_prediction, 'a') as f:
                    for i in range(len(outputs)):
                        f.write('{}\n'.format(outputs[i].item()))
    #         update_list = statistics_softmax(outputs, targets.detach())
    #         statistics_list = update_statistics_list(statistics_list, update_list)
    # mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    # mean_acc, acc_list = calc_acc(statistics_list)
    # return mean_f1_score, f1_score_list, mean_acc, acc_list
    return [], [], [], []

def main(conf):
    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist
    elif conf.dataset == 'UNBC':
        dataset_info = UNBC_pain_infolist

    # data
    train_loader,val_loader,train_data_num,val_data_num = get_dataloader(conf)
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))
    net = RegFullPictureMEFARG(num_classes=conf.num_classes, backbone=conf.arc)

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
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)
