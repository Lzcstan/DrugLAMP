import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from utils import flatten, l2norm, singleton, default
from utils import mask_with_tokens, get_mask_subset_with_prob, prob_mask_like

class SSL(nn.Module):
    def __init__(
        self,
        prot_extractor,
        n_prot_feature,
        *,
        drug_ssl_type = 'simsiam',
        n_hidden = 128,
        **kwargs
    ):
        super().__init__()

        # Prot ssl
        self.extractor = prot_extractor
        self.to_logits = nn.Linear(128, 26 + 1)
        self.llm_to_logits = nn.Linear(n_prot_feature + 1, 26 + 1)

        # Drug ssl
        self.drug_ssl_type = drug_ssl_type
        self.net = SimProj(n_hidden)
        self.llm_net = SimProj(n_hidden)
        if self.drug_ssl_type == 'simsiam':
            self.predictor = PredictorMLP(n_hidden, n_hidden, n_hidden * 4)
        else:
            self.temperature = 0.1

    def drug_simclr(self, vd, xd):
        queries = self.net(rearrange(vd, '... d -> (...) d'))
        keys = self.llm_net(rearrange(xd, '... d -> (...) d'))

        queries, keys = map(flatten, (queries, keys))
        loss = nt_xent_loss(queries, keys, temperature=self.temperature)
        return loss
    
    def drug_simsiam(self, vd, xd):
        drug_one, drug_two = rearrange(vd, '... d -> (...) d'), rearrange(xd, '... d -> (...) d')

        proj_one = self.net(drug_one)
        proj_two = self.llm_net(drug_two)

        pred_one = self.predictor(proj_one)
        pred_two = self.predictor(proj_two)

        with torch.no_grad():
            target_net = self.net
            llm_target_net = self.llm_net
            target_proj_one = target_net(drug_one)
            target_proj_two = llm_target_net(drug_two)
            # target_proj_one.detach_()
            target_proj_one = target_proj_one.detach()
            # target_proj_two.detach_()
            target_proj_two = target_proj_two.detach_()

        loss1 = loss_fn(pred_one, target_proj_two)
        loss2 = loss_fn(pred_two, target_proj_one)
        loss = loss1 + loss2
        return loss.mean()
    
    def prot_mlm(self, seq, extractor, xp, fill_bit, mode, mask_ignore_token_ids={0}, 
                 mask_prob=0.15, replace_prob=0.9, pad_token_id=0, mask_token_id=26):
        no_mask = mask_with_tokens(seq, mask_ignore_token_ids)

        mask = get_mask_subset_with_prob(~no_mask, mask_prob)
        labels = seq.masked_fill(~mask, pad_token_id).long()

        masked_seq = seq.clone().detach()

        replace_prob = prob_mask_like(seq, replace_prob)
        masked_seq = masked_seq.masked_fill(mask * replace_prob, mask_token_id)

        if mode != 'xp':
            embedding = extractor(masked_seq, fill_bit)
            logits = self.to_logits(embedding)
        if mode != 'vp':
            llm_logits = self.llm_to_logits(xp)

        if mode == 'double':
            mlm_loss = (
                F.cross_entropy(
                    logits.transpose(1, 2),
                    labels,
                    ignore_index=pad_token_id
                ) + F.cross_entropy(
                    llm_logits.transpose(1, 2),
                    labels,
                    ignore_index=pad_token_id
                )
            ) / 2
        elif mode == 'vp':
            mlm_loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=pad_token_id)
        else:
            mlm_loss = F.cross_entropy(llm_logits.transpose(1, 2), labels, ignore_index=pad_token_id)
        return mlm_loss
        
    def forward(
        self,
        vp,
        xp,
        fill_bit_p,
        vd,
        xd,
        p_mode='double',
    ):
        prot_ssl_loss = self.prot_mlm(vp, self.extractor, xp, fill_bit_p, p_mode)
        if (vd is None) or (xd is None):
            drug_ssl_loss = 0
        else:
            if self.drug_ssl_type == 'simsiam':
                drug_ssl_loss = self.drug_simsiam(vd, xd)
            else:
                drug_ssl_loss = self.drug_simclr(vd, xd)
        ssl_loss_dict = {
            'prot_ssl': prot_ssl_loss,
            'drug_ssl': drug_ssl_loss,
        }
        return ssl_loss_dict

class SimProj(nn.Module):
    def __init__(self, projection_out, projection_hidden_size = 512):
        super().__init__()
        self.projector = None
        self.projection_out = projection_out
        self.projection_hidden_size = projection_hidden_size

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = SimSiamMLP(dim, self.projection_out, self.projection_hidden_size)
        return projector.to(hidden)

    def forward(self, x):
        projector = self._get_projector(x)
        projection = projector(x)
        return projection
    
def PredictorMLP(dim, proj_out, hidden_size = None):
    hidden_size = default(hidden_size, dim)

    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace = True),
        nn.Linear(hidden_size, proj_out)
    )

def SimSiamMLP(dim, proj_out, hidden_size = 512):
    hidden_size = default(hidden_size, proj_out * 2)

    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias = False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace = True),
        nn.Linear(hidden_size, hidden_size, bias = False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace = True),
        nn.Linear(hidden_size, proj_out, bias = False),
        nn.BatchNorm1d(proj_out, affine = False)
    )

def nt_xent_loss(queries, keys, temperature = 0.1):
    b, device = queries.shape[0], queries.device

    n = b * 2
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)
    logits /= temperature

    labels = torch.cat(((torch.arange(b, device = device) + b - 1), torch.arange(b, device=device)), dim=0)
    loss = F.cross_entropy(logits, labels, reduction = 'sum')
    loss /= n
    return loss

def loss_fn(x, y):
    x = l2norm(x)
    y = l2norm(y)
    return 2 - 2 * (x * y).sum(dim=-1)