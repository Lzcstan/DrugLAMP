import torch
import itertools
import torch.nn as nn

from functools import partial
from utils import tanh_decay, no_decay, cosine_anneal, sigmoid_cosine_distance_p, max_cosine_tanh_decay, l2norm

MARGIN_FN_DICT = {
    'tanh_decay': tanh_decay,
    'no_decay': no_decay,
    'cosine_anneal': cosine_anneal,
    'max_cosine_tanh_decay': max_cosine_tanh_decay
}

def ccpp_p_tri_loss(loss_fn, gt, pid2t, did2t, p_lats, d_lats):
    p_tri_loss = torch.tensor(0.0).to(p_lats[0].device)
    n_tri = 0
    for i, pid in enumerate(pid2t):
        pos_ids = []
        neg_ids = []
        for j, did in enumerate(did2t):
            if gt[pid][did] == 1:
                pos_ids.append(j)
            elif gt[pid][did] == 0:
                neg_ids.append(j)
        
        if len(pos_ids) > 0 and len(neg_ids) > 0:
            n_tri += len(pos_ids) * len(neg_ids)
            aids = [i] * (len(pos_ids) * len(neg_ids))
            pids = []
            nids = []
            for pos_id, neg_id in itertools.product(pos_ids, neg_ids):
                pids.append(pos_id)
                nids.append(neg_id)
            anchor, positive, negative = p_lats[aids], d_lats[pids], d_lats[nids]
            p_tri_loss += loss_fn(anchor, positive, negative)
    
    if n_tri == 0:
        n_tri = 1

    return p_tri_loss / n_tri

class MarginScheduledLossFunction:
    def __init__(
            self,
            loss_script,
            m_ori: float = 0.25,
            n_epoch: int = 100,
            n_re: int = -1,
            update_fn='tanh_decay'
    ):
        self.m_ori = m_ori
        self.n_epoch = n_epoch
        if n_re == -1:
            self.n_re = int(n_epoch * 0.2)
        else:
            self.n_re = n_re

        self._step = 0
        self.m_cur = m_ori

        self._update_fn_str = update_fn
        self._update_m_fn = self._get_update_fn(update_fn)

        self._loss_script = loss_script
        self._update_loss_fn()

    @property
    def margin(self):
        return self.m_cur
    
    def _get_update_fn(self, fn_str):
        return partial(MARGIN_FN_DICT[fn_str], self.m_ori, self.n_re)
    
    def _update_loss_fn(self):
        self._loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=sigmoid_cosine_distance_p,
            margin=self.margin,
            reduction='sum' # n_tri
        )
    
    def step(self):
        self._step += 1
        if self._step == self.n_re:
            self.reset()
        else:
            self.m_cur = self._update_m_fn(self._step)
            self._update_loss_fn()

    def reset(self):
        self._step = 0
        self.m_cur = self._update_m_fn(self._step)
        self._update_loss_fn()
    
    def __call__(self, **kwargs):
        return self._loss_script(self._loss_fn, **kwargs)

class CrossModality(nn.Module):
    def __init__(
        self,
        *,
        use_cm = True,
        hidden_size = 128,
        max_margin=0.5,
        n_re=100,
        **kwargs
    ):
        self.use_cm = use_cm
        super().__init__()
        self.prot2latent = Mean2Embed(hidden_size)
        self.aug_prot2latent = Mean2Embed(hidden_size)
        self.drug2latent = Mean2Embed(hidden_size)
        self.aug_drug2latent = Mean2Embed(hidden_size)

        self.to_prot_latent = nn.Linear(hidden_size * 2, hidden_size * 2, bias = False)
        self.to_drug_latent = nn.Linear(hidden_size * 2, hidden_size * 2, bias = False)

        self.m_sch_loss_fn = MarginScheduledLossFunction(ccpp_p_tri_loss, m_ori=max_margin, n_re=n_re)

    def step(self):
        self.m_sch_loss_fn.step()

    def forward(
        self,
        prot,
        aug_prot,
        drug,
        aug_drug,
        meta
    ):
        # construct gt_mat
        pid2t = {meta[t]['Prot_ID']: t for t in range(len(meta))}
        did2t = {meta[t]['Drug_ID']: t for t in range(len(meta))}
        default_cell = 0 if self.use_cm else -1
        gt_mat = {pid: {did: default_cell for did in did2t.keys()} for pid in pid2t.keys()}
        
        for m in meta:
            gt_mat[m['Prot_ID']][m['Drug_ID']] = int(m['Y'])

        # construct prot, aug_prot, drug, aug_drug
        prot = prot[list(pid2t.values())]
        aug_prot = aug_prot[list(pid2t.values())]
        drug = drug[list(did2t.values())]
        aug_drug = aug_drug[list(did2t.values())]

        prot_embed = self.prot2latent(prot.mean(dim=1))
        aug_prot_embed = self.aug_prot2latent(aug_prot.mean(dim=1))
        drug_embed = self.drug2latent(drug.mean(dim=1))
        aug_drug_embed = self.aug_drug2latent(aug_drug.mean(dim=1))

        prot_embeds = torch.cat([prot_embed, aug_prot_embed], dim=-1)
        drug_embeds = torch.cat([drug_embed, aug_drug_embed], dim=-1)

        prot_lats = self.to_prot_latent(prot_embeds)
        drug_lats = self.to_drug_latent(drug_embeds)
        prot_lats, drug_lats = map(l2norm, (prot_lats, drug_lats))

        return self.m_sch_loss_fn(gt=gt_mat, pid2t=pid2t, did2t=did2t, p_lats=prot_lats, d_lats=drug_lats)

def Mean2Embed(hidden=128):
    return nn.Sequential(
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden)
    )

