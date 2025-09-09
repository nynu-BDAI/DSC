
# ------------------------------------------------------------------------
# Modified from UniMoCo (https://github.com/dddzg/unimoco)
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Definition of the Supervised Contrastive Loss
"""
from torch import nn
import torch
class SupContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(SupContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):

        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1).unsqueeze(1)  #所有负样本（相对于锚点）的指数相似度之和，希望这个值更小
        sum_pos = (y_true * torch.exp(-y_pred)) #希望这个值小，这样意味着-y_pred很大！
        num_pos = y_true.sum(1) #正样本个数
        loss = torch.log(1 + sum_neg * sum_pos).sum(1) / num_pos

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss