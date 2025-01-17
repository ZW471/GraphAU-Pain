from math import cos, pi
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def statistics(pred, y, thresh):
    batch_size = pred.size(0)
    class_nb = pred.size(1)
    pred = pred >= thresh
    pred = pred.long()
    statistics_list = []
    for j in range(class_nb):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(batch_size):
            if pred[i][j] == 1:
                if y[i][j] >= 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                else:
                    raise ValueError(f'Unexpected y value {y[i][j]}')
            elif pred[i][j] == 0:
                if y[i][j] >= 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                else:
                    raise ValueError(f'Unexpected y value {y[i][j]}')
            else:
                raise ValueError(f'Unexpected target value {pred[i][j]}')
        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
    return statistics_list

def statistics_softmax(pred, y):
    batch_size = pred.size(0)
    class_nb = pred.size(1)
    cl_index = torch.argmax(pred, dim=1, keepdim=True)  # Get index of max probability
    pred = torch.zeros_like(pred).scatter_(1, cl_index, 1)  # Create hard label tensor
    pred = pred.long()
    statistics_list = []
    for j in range(class_nb):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(batch_size):
            if pred[i][j] == 1:
                if y[i][j] >= 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                else:
                    raise ValueError(f'Unexpected y value {y[i][j]}')
            elif pred[i][j] == 0:
                if y[i][j] >= 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                else:
                    raise ValueError(f'Unexpected y value {y[i][j]}')
            else:
                raise ValueError(f'Unexpected target value {pred[i][j]}')
        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
    return statistics_list




def calc_f1_score(statistics_list):
    f1_score_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']

        precise = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)

    return mean_f1_score, f1_score_list


def calc_acc(statistics_list):
    acc_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']
        TN = statistics_list[i]['TN']

        acc = (TP+TN)/(TP+TN+FP+FN)
        acc_list.append(acc)
    mean_acc_score = sum(acc_list) / len(acc_list)

    return mean_acc_score, acc_list


def update_statistics_list(old_list, new_list):
    if not old_list:
        return new_list

    assert len(old_list) == len(new_list)

    for i in range(len(old_list)):
        old_list[i]['TP'] += new_list[i]['TP']
        old_list[i]['FP'] += new_list[i]['FP']
        old_list[i]['TN'] += new_list[i]['TN']
        old_list[i]['FN'] += new_list[i]['FN']

    return old_list


def BP4D_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU10: {:.2f} AU12: {:.2f} AU14: {:.2f} AU15: {:.2f} AU17: {:.2f} AU23: {:.2f} AU24: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9],100.*list[10],100.*list[11])}
    return infostr

def DISFA_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f}  AU6: {:.2f} AU9: {:.2f} AU12: {:.2f}  AU25: {:.2f} AU26: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7])}
    # infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU5: {:.2f} AU6: {:.2f} AU9: {:.2f} AU12: {:.2f} AU15: {:.2f} AU17: {:.2f} AU20: {:.2f} AU25: {:.2f} AU26: {:.2f}'.format(100.*list[0], 100.*list[1], 100.*list[2], 100.*list[3], 100.*list[4], 100.*list[5], 100.*list[6], 100.*list[7], 100.*list[8], 100.*list[9], 100.*list[10], 100.*list[11])}
    return infostr

def UNBC_infolist(list, use_disfa=True):
    # infostr = {'AU4: {:.2f} AU6/7: {:.2f} AU9/10: {:.2f} AU43: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3])}
    # infostr = {'AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU9: {:.2f} AU10: {:.2f} AU12: {:.2f} AU20: {:.2f} AU25: {:.2f} AU26: {:.2f} AU43: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9])}
    # disfa processed
    if use_disfa:
        infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f}  AU6: {:.2f} AU9: {:.2f} AU12: {:.2f}  AU25: {:.2f} AU26: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7])}
    else:
        infostr = {'AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU9: {:.2f} AU10: {:.2f} AU12: {:.2f} AU20: {:.2f} AU25: {:.2f} AU26: {:.2f} AU43: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9])}
    return infostr

def UNBC_pain_infolist(list):

    infostr = {'No Pain: {:.2f} Mild Pain: {:.2f} Pain: {:.2f}'.format(100.*list[0],100.*list[1],100.*list[2])}
    return infostr

def UNBC_pain_infolist_binary(list):

    infostr = {'No Pain: {:.2f} Pain: {:.2f}'.format(100.*list[0],100.*list[1])}
    return infostr


def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter):

    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter
    lr = init_lr * (1 + cos(pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class image_train(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img, flip, offset_x, offset_y):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            PlaceCrop(self.crop_size, offset_x, offset_y),
            SetFlip(flip),
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


class image_test(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=torch.device('cpu'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model


class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight

    def forward(self, x, y):

        # x = x[:, 2:]
        # y = y[:, 2:]

        if self.weight is not None:
            self.weight = self.weight.to(x.device)

        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)
            # loss = loss * self.weight[2:].view(1,-1)

        loss = loss.mean(dim=-1)
        return -loss.mean()

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        """
        Weighted Cross-Entropy Loss for multi-class classification.

        Args:
            eps (float): Small constant to avoid log(0).
            disable_torch_grad (bool): If True, torch gradient computations will be disabled (not typically needed for PyTorch loss functions).
            weight (Tensor or None): Class weights. Tensor of shape [num_classes].
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight

    def forward(self, x, y):
        """
        Compute the weighted cross-entropy loss.

        Args:
            x (Tensor): Logits of shape [batch_size, num_classes].
            y (Tensor): Ground truth labels of shape [batch_size, num_classes] (one-hot encoded).

        Returns:
            Tensor: Scalar loss value.
        """
        if self.weight is not None:
            self.weight = self.weight.to(x.device)
        # Apply softmax to logits to get class probabilities
        probs = nn.functional.softmax(x, dim=-1).clamp(min=self.eps)

        # Compute log probabilities
        log_probs = torch.log(probs)

        # Weighted cross-entropy loss
        loss = -y * log_probs

        if self.weight is not None:
            # Apply class weights
            loss = loss * self.weight.view(1, -1)

        # Average loss over classes and batch
        loss = loss.sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss for classification tasks.

        Args:
            alpha (float): Weighting factor for class imbalance (optional).
            gamma (float): Focusing parameter for hard examples.
            reduction (str): Specifies reduction to apply to loss ('none', 'mean', or 'sum').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Forward pass for Focal Loss.

        Args:
            logits (Tensor): Predicted logits of shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels (for binary: one-hot or 0/1, for multi-class: class indices).

        Returns:
            Calculated Focal Loss.
        """
        # Compute probabilities using softmax for multi-class or sigmoid for binary
        if logits.shape[-1] > 1:  # Multi-class
            probs = F.softmax(logits, dim=-1)
        else:  # Binary
            probs = torch.sigmoid(logits)

        # Compute the focal loss
        pt = probs * targets + (1 - probs) * (1 - targets)  # p_t
        logpt = torch.log(pt.clamp(min=1e-8))  # log(p_t)
        focal_loss = - self.alpha * ((1 - pt) ** self.gamma) * logpt

        # Reduce the loss (mean, sum, or return as-is)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class WeightedMSELoss(nn.Module):
    def __init__(self, weight=None):
        """
        Initialize the WeightedMSELoss.

        Args:
            weight (torch.Tensor or None): Tensor of weights for each feature. If None, no weighting is applied.
        """
        super(WeightedMSELoss, self).__init__()
        self.weight = weight

    def forward(self, x, y):
        """
        Compute the weighted mean squared error between x and y.

        Args:
            x (torch.Tensor): Predicted values, shape (batch_size, num_features).
            y (torch.Tensor): Ground truth values, shape (batch_size, num_features).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Element-wise squared error
        squared_error = (x * 5 - y) ** 2

        # Apply weights if provided
        if self.weight is not None:
            weighted_error = squared_error * self.weight.view(1, -1)
        else:
            weighted_error = squared_error

        # Mean over features
        loss_per_sample = weighted_error.mean(dim=-1)

        # Mean over batch
        return loss_per_sample.mean()