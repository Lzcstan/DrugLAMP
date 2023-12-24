# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from model.PMMA.attention import Attention
from model.PMMA.embed import Embeddings 
from model.PMMA.mlp import Mlp
from model.PMMA.block import PMMABlock

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer_with_mol = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size * 2, eps=1e-6)
        for i in range(config.transformer["num_p_plus_s_layers"]):
            if i < 2:
                layer_with_mol = PMMABlock(config, vis, mm=True)
            else:
                if i == 2:
                    config.hidden_size = config.hidden_size * 2                
                layer_with_mol = PMMABlock(config, vis)
            self.layer_with_mol.append(copy.deepcopy(layer_with_mol))

    def forward(self, hidden_states, mol=None):
        attn_weights = []
        guided_attn_weights = []
        
        for (i, layer_block) in enumerate(self.layer_with_mol):
            if i>=2:
                if i == 2:
                    hidden_states = torch.cat((hidden_states, mol), dim=-1)
                hidden_states, weights, guided_weights = layer_block(hidden_states)
            else:
                hidden_states, mol, weights, guided_weights = layer_block(hidden_states, mol)
            if self.vis:
                attn_weights.append(weights)
                guided_attn_weights.append(guided_weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights, guided_attn_weights

class LinAttnEncoder(nn.Module):
    def __init__(self, config, vis):
        super(LinAttnEncoder, self).__init__()
        self.vis = vis
        self.layer_with_mol = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size * 2, eps=1e-6)
        for i in range(config.transformer["num_p_plus_s_layers"]):
            if i < 2:
                layer_with_mol = PMMABlock(config, vis, mm=True)
            else:
                layer_with_mol = MultiHeadLinearAttention(d_model=config.hidden_size * 2, d_diff=config.hidden_size * 8, nhead=8, dropout=config.dropout, activation='gelu')

            self.layer_with_mol.append(copy.deepcopy(layer_with_mol))

    def forward(self, hidden_states, mol=None):
        attn_weights = []
        guided_attn_weights = []
        
        for (i, layer_block) in enumerate(self.layer_with_mol):
            if i>=2:
                if i == 2:
                    hidden_states = torch.cat((hidden_states, mol), dim=-1)
                h = hidden_states
                hidden_states = layer_block(hidden_states)
                hidden_states = hidden_states + h
            else:
                hidden_states, mol, weights, guided_weights = layer_block(hidden_states, mol)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights, guided_attn_weights
    
class MultiHeadLinearAttention(nn.Module):
    def __init__(self, d_model, nhead, d_diff=32, dropout=0.1, activation='tanh'):
        super(MultiHeadLinearAttention, self).__init__()
        if activation == 'tanh':
            # self.act = torch.tanh
            self.act = nn.Tanh()
        elif activation == 'relu':
            # self.act = F.relu
            self.act = nn.ReLU()
        elif activation == 'gelu':
            # self.act = F.gelu
            self.act = nn.GELU()
        else:
            raise NotImplementedError
        
        self.lin1 = nn.Linear(d_model, d_diff)
        self.lin2 = nn.Linear(d_diff, nhead)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def softmax(self, input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)

    def forward(self, v):
        attn = self.dropout1(self.act(self.lin1(v)))
        attn = self.dropout2(self.lin2(attn))
        attn = self.softmax(attn, 1)
        attn = attn.transpose(1, 2) # (B, nhead, len_seq)

        H = attn.size()[1]
        B, L, embed_dim = v.size()
        head_dim = embed_dim // H
        v = v.contiguous().view(B * H, L, head_dim)
        attn = attn.contiguous().view(B * H, L).unsqueeze(-1)
        v = attn * v
        v = v.contiguous().view(B, L, embed_dim)
        return v