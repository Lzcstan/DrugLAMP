import torch
import torch.nn as nn
import torch.nn.functional as F

from dgllife.model.gnn import GCN
from configs import get_model_defaults
from model.PGCA import GuidedCrossAttention
from model.cross_modality import CrossModality
from model.self_supervised_learning import SSL
from model.PMMA import MultiHeadLinearAttention, PairedMultimodelAttention

CONFIGS = {
    'LAMP': get_model_defaults,
}

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels.float())
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)
    
class DrugLAMPBase(nn.Module):
    def __init__(self, n_drug_feature, n_prot_feature, n_hidden=128, **cfg):
        super(DrugLAMPBase, self).__init__()
        drug_padding = cfg["DRUG"]["PADDING"]
        drug_in_feats = cfg["DRUG"]["NODE_IN_FEATS"]
        self.site_len = cfg['PROTEIN']['SITE_LEN']
        self.seq_len_q = cfg['PROTEIN']['SEQ_LEN']
        protein_padding = cfg["PROTEIN"]["PADDING"]
        protein_kernel_size = cfg["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = cfg["DECODER"]["IN_DIM"]
        mlp_binary = cfg["DECODER"]["BINARY"]
        mlp_out_dim = cfg["DECODER"]["OUT_DIM"]
        mlp_hidden_dim = cfg["DECODER"]["HIDDEN_DIM"]
        drug_embedding = n_hidden
        drug_hidden_feats = [n_hidden] * 3
        protein_emb_dim = n_hidden
        protein_num_filters = [n_hidden] * 3

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinCNN(protein_emb_dim, protein_num_filters, 
                                            protein_kernel_size, protein_padding)

        # SSL
        self.ssl_model = SSL(
            prot_extractor=self.protein_extractor,
            n_prot_feature=n_prot_feature,
            drug_ssl_type='simsiam',
            n_hidden=n_hidden
        )

        # CCPP
        self.ccpp = CrossModality(
            use_cm=True,
            hidden_size=n_hidden,
            max_margin=cfg["RS"]["MAX_MARGIN"],
            n_re=cfg["RS"]["RESET_EPOCH"]
        )

        # LAMP config
        model_cfg = CONFIGS['LAMP'](n_hidden)

        # Drug branch
        self.lin_d1 = nn.Linear(n_drug_feature + 1, 2 * n_hidden)
        self.act_d = nn.GELU()
        self.d_norm = nn.LayerNorm(2 * n_hidden)
        self.lin_d2 = nn.Linear(2 * n_hidden, n_hidden)

        # Prot branch
        self.p_adaptor_wo_skip_connect = FeedForwardLayer(n_prot_feature + 1, n_hidden)
        self.lin_p1 = nn.Linear(n_prot_feature + 1, 2 * n_hidden)
        self.act_p = nn.GELU()
        self.p_norm = nn.LayerNorm(2 * n_hidden)
        self.lin_p2 = nn.Linear(2 * n_hidden, n_hidden)

        self.v_gca = GuidedCrossAttention(embed_dim=n_hidden, num_heads=1)
        self.v_mhla = MultiHeadLinearAttention(d_model=n_hidden * 2, d_diff=n_hidden * 8, nhead=8, dropout=model_cfg.mlha_dropout, activation='gelu')
        self.v_gca_norm = nn.LayerNorm(n_hidden * 2)
        self.x_gca = GuidedCrossAttention(embed_dim=n_hidden, num_heads=1)
        self.x_mhla = MultiHeadLinearAttention(d_model=n_hidden * 2, d_diff=n_hidden * 8, nhead=8, dropout=model_cfg.mlha_dropout, activation='gelu')
        self.x_gca_norm = nn.LayerNorm(n_hidden * 2)

        self.pmma = PairedMultimodelAttention(config=model_cfg, vis=False)
        self.mlp_classifier = MLP(mlp_in_dim * 2, mlp_hidden_dim * 2, mlp_out_dim * 2, binary=mlp_binary)

    def get_cross_attn_mat(self, modality='v'):
        if modality == 'v':
            self.A_v_gca = self.A_v_gca.cpu()
            return self.A_v_gca
        else:
            self.A_x_gca = self.A_x_gca.cpu()
            return self.A_x_gca
        
    def get_inter_attn_mat(self):
        return self.attn, self.guide_attn

    def forward(self, vd, vp, xd, xp, mode="train"):
        pass
        
class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats) # Tid: Node feats variable
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats
    
class ProteinCNN(nn.Module): # Tid: add fill bit
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26 + 1, embedding_dim - 1, padding_idx=0) # Tid: add fill bit, fit mlm
        else:
            self.embedding = nn.Embedding(26 + 1, embedding_dim - 1) # Tid: add fill bit, fit mlm
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0], padding='same')
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1], padding='same')
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2], padding='same')
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v, fill_mask):
        v = self.embedding(v.long())
        v = torch.cat((v, fill_mask.unsqueeze(-1)), dim=-1)
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v))) # -2
        v = self.bn2(F.relu(self.conv2(v))) # -5
        v = self.bn3(F.relu(self.conv3(v))) # -8
        v = v.view(v.size(0), v.size(2), -1)
        return v
    
class FeedForwardLayer(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_h)
        self.lin2 = nn.Linear(d_h, d_in)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_h)

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.norm(x)
        x = self.lin2(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(self.act1(self.fc1(x)))
        x = self.bn2(self.act2(self.fc2(x)))
        x = self.bn3(self.act3(self.fc3(x)))
        x = self.fc4(x)
        return x