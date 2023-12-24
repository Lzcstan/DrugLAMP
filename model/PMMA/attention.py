import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math

class Attention(nn.Module):
    def __init__(self, config, vis, mm=True):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attn_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attn_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        if mm:
            self.query_mol = Linear(config.hidden_size, self.all_head_size)
            self.key_mol = Linear(config.hidden_size, self.all_head_size)
            self.value_mol = Linear(config.hidden_size, self.all_head_size)
            self.out_mol = Linear(config.hidden_size, config.hidden_size)
            self.attn_dropout_mol = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_pm = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_mp = Dropout(config.transformer["attention_dropout_rate"])
            self.proj_dropout_mol = Dropout(config.transformer["attention_dropout_rate"])
            # self.fc = Linear(config.mol_len + config.feat_len + 1, config.feat_len + 1)
            # self.fc_mol = Linear(config.mol_len + config.feat_len + 1, config.mol_len)
            self.fc = Linear(config.hidden_size * 2, config.hidden_size) # Tid
            self.fc_mol = Linear(config.hidden_size * 2, config.hidden_size) # Tid         

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # print(self.num_attention_heads, self.attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def paired_attention(self, q, k, v, q_p, label='mol'):
        if label == 'mol':
            attn_dropout = self.attn_dropout_mol
            attn_dropout_p = self.attn_dropout_pm
            fc = self.fc_mol
            out = self.out_mol
            proj_dropout = self.proj_dropout_mol
        else:
            attn_dropout = self.attn_dropout
            attn_dropout_p = self.attn_dropout_mp
            fc = self.fc
            out = self.out
            proj_dropout = self.proj_dropout

        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn / math.sqrt(self.attn_head_size)
        attn = self.softmax(attn)
        if label == 'prot':
            weights = attn if self.vis else None # Tid: no use
        attn = attn_dropout(attn)
        attn = torch.matmul(attn, v)
        attn = attn.permute(0, 2, 1, 3).contiguous()
        new_attn_shape = attn.size()[: -2] + (self.all_head_size, )
        attn = attn.view(*new_attn_shape)

        attn_p = torch.matmul(q_p, k.transpose(-1, -2))
        attn_p = attn_p / math.sqrt(self.attn_head_size)
        attn_p = self.softmax(attn_p)
        if label == 'prot':
            guided_weights = attn_p if self.vis else None
        attn_p = attn_dropout_p(attn_p)
        attn_p = torch.matmul(attn_p, v)
        attn_p = attn_p.permute(0, 2, 1, 3).contiguous()
        new_attn_shape = attn_p.size()[: -2] + (self.all_head_size, )
        attn_p = attn_p.view(*new_attn_shape)
        
        # attn = fc(torch.concat((attn, attn_p), dim=1).permute(0, 2, 1)).permute(0, 2, 1)
        attn = fc(torch.cat((attn, attn_p), dim=-1)) # Tid
        attn = out(attn)
        attn = proj_dropout(attn)

        if label == 'mol':
            return attn
        else:
            return attn, weights, guided_weights

    def forward(self, hidden_states, mol=None):
        mixed_q_layer = self.query(hidden_states) # B, L, H * head_size (hidden_size)
        mixed_k_layer = self.key(hidden_states)
        mixed_v_layer = self.value(hidden_states)

        if mol is not None:
            mol_q = self.query_mol(mol)
            mol_k = self.key_mol(mol)      
            mol_v = self.value_mol(mol)      

        q_layer = self.transpose_for_scores(mixed_q_layer) # B, H, L, head_size
        k_layer = self.transpose_for_scores(mixed_k_layer)
        v_layer = self.transpose_for_scores(mixed_v_layer)

        if mol is not None:
            q_layer_mol = self.transpose_for_scores(mol_q)
            k_layer_mol = self.transpose_for_scores(mol_k)
            v_layer_mol = self.transpose_for_scores(mol_v)

        if mol is None: # Tid: TODO-reduce the use of memory
            attn = torch.matmul(q_layer, k_layer.transpose(-1, -2)) # B, H, L, L
            attn = attn / math.sqrt(self.attn_head_size) # B, H, L, L
            attn = self.softmax(attn) # B, H, L, L
            weights = attn if self.vis else None
            attn = self.attn_dropout(attn)

            attn = torch.matmul(attn, v_layer) # B, H, L, head_size
            attn = attn.permute(0, 2, 1, 3).contiguous() # B, L, H, head_size
            new_attn_shape = attn.size()[:-2] + (self.all_head_size,)
            attn = attn.view(*new_attn_shape) # B, L, H * head_size (hidden_size)
            attn = self.out(attn)
            attn = self.proj_dropout(attn)
            return attn, weights, None
        else:
            attn_prot, weights, guided_weights = self.paired_attention(q_layer, k_layer, v_layer, q_layer_mol, label='prot')
            attn_mol = self.paired_attention(q_layer_mol, k_layer_mol, v_layer_mol, q_layer)
 
            return attn_prot, attn_mol, weights, guided_weights