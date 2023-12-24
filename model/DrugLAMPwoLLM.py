import torch
from .basic_model import DrugLAMPBase

class DrugLAMPwoLLM(DrugLAMPBase):
    def __init__(self, n_drug_feature, n_prot_feature, n_hidden=128, **cfg):
        super().__init__(n_drug_feature, n_prot_feature, n_hidden, **cfg)

    def forward(self, vd, vp, xd, xp, mode="train"):
        vd = self.drug_extractor(vd) # (64, 512, 128)

        fill_bit_p = torch.zeros_like(xp.sum(dim=-1))
        mask_p = xp.sum(dim=-1) == 0
        fill_bit_p[mask_p] = 1

        vp_ssl = vp
        xp_ssl = None
        fill_bit_p_ssl = fill_bit_p
        vd_ssl = vd
        xd_ssl = xd
        ssl = {
            'vp': vp_ssl,
            'xp': xp_ssl,
            'fill_bit_p': fill_bit_p_ssl,
            'vd': vd_ssl,
            'xd': xd_ssl,
            'p_mode': 'vp',
        }

        vp = self.protein_extractor(vp, fill_bit_p)
        site_seq_len = self.seq_len_q // self.site_len
        vp = vp.view(-1, self.site_len, site_seq_len, vp.size()[-1])
        vp = torch.mean(vp, dim=1)

        # PGD GCA
        mv, self.A_v_gca = self.v_gca(vp.permute(1, 0, 2), vd.permute(1, 0, 2), vd.permute(1, 0, 2))
        mv = mv.permute(1, 0, 2)

        mv = torch.cat((vp, mv), 2)

        hv = mv
        mv = self.v_mhla(hv)
        mv = mv + hv

        mv = self.v_gca_norm(mv)

        f, self.attn, self.guide_attn = self.pmma(mv, mv)
        f = torch.mean(f, dim=1)
        score = self.mlp_classifier(f) # (B, 256) -> (B, 1)
        if mode == "train":
            return vd, vp, ssl, None, score
        elif mode == "eval":
            return vd, vp, score, self.attn