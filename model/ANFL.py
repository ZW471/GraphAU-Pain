import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .resnet import resnet18, resnet50, resnet101
from .graph import normalize_digraph
from .basic_block import *


class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels,self.in_channels)
        self.V = nn.Linear(self.in_channels,self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == 'dots':
            si = x.detach()
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'cosine':
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'l1':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x


class Head(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.gnn = GNN(self.in_channels, self.num_classes,neighbor_num=neighbor_num,metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        # FGG
        f_v = self.gnn(f_v)
        b, n, c = f_v.shape  # b: batch size, n: num of nodes (AUs), c: dim of node feature
        sc = self.sc
        sc = self.relu(sc) # sc: self-connection
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)  # cl: class label
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        return cl


class HeadPEAU(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(HeadPEAU, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers.append(layer)
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.gnn = GNN(self.in_channels, self.num_classes, neighbor_num=neighbor_num, metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)  # Aggregated feature representation

        # FGG
        f_v = self.gnn(f_v)

        # Extract dimensions
        b, n, c = f_v.shape  # b: batch size, n: num of nodes (AUs), c: dim of node feature

        # Self-connection
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)

        # Class label calculation
        cl = F.normalize(f_v, p=2, dim=-1)  # Normalizing GNN output
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)  # Computing class logits

        # **PE Score**
        # Obtain the PE score by performing global sum pooling over the node features in the GNN output
        pe_score = f_v.sum(dim=1)  # Sum pooling along the node dimension

        return cl, pe_score

class MEFARG(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(MEFARG, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_classes, neighbor_num, metric)

    def forward(self, x):
        # x: b d c
        x = self.backbone(x)
        x = self.global_linear(x)
        cl = self.head(x)
        return cl


class HeadBackboneOnly(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(HeadBackboneOnly, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc = nn.Linear(36 * 2048, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.flatten(1)
        cl = self.fc(x)
        return cl

class BackboneOnly(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(BackboneOnly, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = HeadBackboneOnly(self.out_channels, num_classes, neighbor_num, metric)

    def forward(self, x):
        # x: b d c
        x = self.backbone(x)
        cl = self.head(x)
        return cl


class FullPictureMEFARG(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base', neighbor_num=4, metric='dots', binary=False):
        super(FullPictureMEFARG, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None
        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = HeadPEAU(self.out_channels, num_classes, neighbor_num, metric)

        self.fc_au = nn.Linear(8, 36)  # Fully connected layer for pain intensity mapping
        self.fc_bb = nn.Linear(2048, 36)  # Fully connected layer for pain intensity mapping
        self.fc_pe = nn.Linear(512, 36)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        if binary:
            self.fc = nn.Linear(72, 2)
        else:
            self.fc = nn.Linear(72, 3)  # Fully connected layer for pain intensity mapping
    def forward(self, x):
        # x: b d c
        bb = self.backbone(x)
        x = self.global_linear(bb)

        bb = self.fc_bb(bb)
        bb = self.relu(bb)

        au, pe = self.head(x)
        au = self.fc_au(au)
        au = self.relu(au)
        au = au.unsqueeze(1)

        pe = self.fc_pe(pe)
        pe = self.relu(pe)

        cl = torch.matmul(au, bb)
        cl = cl.squeeze(1)
        cl = self.relu(cl)
        cl = torch.cat((cl, pe), dim=1)
        cl = self.fc(cl)
        return cl

class AuAndPeMEFARG(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base', neighbor_num=4, metric='dots', binary=False):
        super(AuAndPeMEFARG, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None
        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = HeadPEAU(self.out_channels, num_classes, neighbor_num, metric)

        self.fc_au = nn.Linear(8, 32)  # Fully connected layer for pain intensity mapping
        self.fc_bb = nn.Linear(2048, 32)  # Fully connected layer for pain intensity mapping
        self.fc_pe = nn.Linear(512, 36)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(72, 32),
            nn.ReLU(),
            nn.Linear(32, 2 if binary else 3)
        )
    def forward(self, x):
        # x: b d c
        bb = self.backbone(x)
        x = self.global_linear(bb)

        bb = self.fc_bb(bb)
        bb = self.relu(bb)

        au, pe = self.head(x)
        au = self.fc_au(au)
        au = self.relu(au)
        au = au.unsqueeze(1)

        pe = self.fc_pe(pe)
        pe = self.relu(pe)

        bb = bb.transpose(1, 2)  # Assuming bb has dimensions [batch_size, features, another_dim]
        cl = torch.matmul(au, bb)
        cl = cl.squeeze(1)
        cl = self.relu(cl)
        cl = torch.cat((cl, pe), dim=1)
        cl = self.fc(cl)
        return cl

class RegFullPictureMEFARG(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(RegFullPictureMEFARG, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_classes, neighbor_num, metric)

        self.fc_au = nn.Linear(8, 36)  # Fully connected layer for pain intensity mapping
        self.fc_bb = nn.Linear(2048, 36)  # Fully connected layer for pain intensity mapping
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(36, 1)  # Fully connected layer for pain intensity mapping
    def forward(self, x):
        # x: b d c
        bb = self.backbone(x)
        x = self.global_linear(bb)

        bb = self.fc_bb(bb)
        bb = self.relu(bb)

        au = self.head(x)
        au = self.fc_au(au)
        au = self.relu(au)
        au = au.unsqueeze(1)

        cl = torch.matmul(au, bb)
        cl = cl.squeeze(1)
        cl = self.relu(cl)
        cl = self.fc(cl)
        intensity = self.sigmoid(cl) * 16
        return intensity

class HeadNoGNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(HeadNoGNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.gnn = GNN(self.in_channels, self.num_classes,neighbor_num=neighbor_num,metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        # FGG
        # f_v = self.gnn(f_v)
        b, n, c = f_v.shape  # b: batch size, n: num of nodes (AUs), c: dim of node feature
        sc = self.sc
        sc = self.relu(sc) # sc: self-connection
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)  # cl: class label
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        return cl

class FullPictureMEFARGNoGNN(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(FullPictureMEFARGNoGNN, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = HeadNoGNN(self.out_channels, num_classes, neighbor_num, metric)
        self.fc_au = nn.Linear(8, 36)  # Fully connected layer for pain intensity mapping
        self.fc_bb = nn.Linear(2048, 36)  # Fully connected layer for pain intensity mapping
        self.relu = nn.ReLU()
        self.fc = nn.Linear(36, 3)  # Fully connected layer for pain intensity mapping
    def forward(self, x):
        # x: b d c
        bb = self.backbone(x)
        x = self.global_linear(bb)

        bb = self.fc_bb(bb)
        bb = self.relu(bb)

        au = self.head(x)
        au = self.fc_au(au)
        au = self.relu(au)
        au = au.unsqueeze(1)

        cl = torch.matmul(au, bb)
        cl = cl.squeeze(1)
        cl = self.fc(cl)
        return cl

class HeadPE(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(HeadPE, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.gnn = GNN(self.in_channels, self.num_classes,neighbor_num=neighbor_num,metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.fc = nn.Linear(self.num_classes, 1)  # Fully connected layer for pain intensity mapping
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        # FGG
        f_v = self.gnn(f_v)
        b, n, c = f_v.shape  # b: batch size, n: num of nodes (AUs), c: dim of node feature
        sc = self.sc
        sc = self.relu(sc) # sc: self-connection
        sc = F.normalize(sc, p=2, dim=-1)
        cl_au = F.normalize(f_v, p=2, dim=-1)  # cl: class label (AU), need to replace this with pain level
        cl_au = (cl_au * sc.view(1, n, c)).sum(dim=-1)

        pain_intensity = self.fc(cl_au)
        pain_intensity = self.relu(pain_intensity)
        return pain_intensity.squeeze(-1)  # Final pain intensity

class PainEstimation(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(PainEstimation, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = HeadPE(self.out_channels, num_classes, neighbor_num, metric)

    def forward(self, x):
        # x: b d c
        x = self.backbone(x)
        x = self.global_linear(x)
        pain_intensity = self.head(x)
        return pain_intensity * 16