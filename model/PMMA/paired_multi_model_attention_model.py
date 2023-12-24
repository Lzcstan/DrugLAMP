# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

from model.PMMA.embed import Embeddings 
from model.PMMA.encoder import Encoder
import torch.nn.functional as F


class PairedMultimodelAttention(nn.Module):
    def __init__(self, config, vis=True):
        super(PairedMultimodelAttention, self).__init__()
        self.embeddings = Embeddings(config, mol_len=config.mol_len)
        self.encoder = Encoder(config, vis)
        # self.fill_seq = nn.Parameter(torch.zeros(1, 256 - config.mol_len[mol_type]))

    def forward(self, prot, mol=None):
        # if mol != None:
            # fill_seq = self.fill_seq.expand(mol.shape[0], -1)
            # mol = torch.cat((fill_seq, mol), dim=1)
            # mol = mol.view(mol.shape[0], -1, 2)
        embedding_output, mol = self.embeddings(prot, mol) # (b, feat_len * 3, d_h) + (b, mol_len * 3, d_h) -> (b, feat_len * 3 + 1, d_h) + (b, mol_len * 3, d_h)
        encoded, attn_weights, guided_attn_weights = self.encoder(embedding_output, mol)
        return encoded, attn_weights, guided_attn_weights
    

class FocalLossV1(nn.Module):
    
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss