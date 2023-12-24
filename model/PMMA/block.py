# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

from model.PMMA.attention import Attention
from model.PMMA.mlp import Mlp


class PMMABlock(nn.Module):
    def __init__(self, config, vis, mm=False):
        super(PMMABlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        if mm:
            self.att_norm_mol = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_norm_mol = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_mol = Mlp(config)

        self.ffn = Mlp(config)
        self.attn = Attention(config, vis, mm)

    def forward(self, prot, mol=None):
        if mol is None:
            h = prot
            prot = self.attention_norm(prot)
            prot, weights, guided_weights = self.attn(prot)
            prot = prot + h

            h = prot
            prot = self.ffn_norm(prot)
            prot = self.ffn(prot)
            prot = prot + h
            return prot, weights, guided_weights
        else:
            h = prot
            h_mol = mol
            prot = self.attention_norm(prot)
            mol = self.att_norm_mol(mol)
            prot, mol, weights, guided_weights = self.attn(prot, mol) 
            prot = prot + h
            mol = mol + h_mol

            h = prot
            h_mol = mol
            prot = self.ffn_norm(prot)
            mol = self.ffn_norm_mol(mol)
            prot = self.ffn(prot)
            mol = self.ffn_mol(mol)
            prot = prot + h
            mol = mol + h_mol
            return prot, mol, weights, guided_weights