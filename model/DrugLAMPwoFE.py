import torch
from .basic_model import DrugLAMPBase

class DrugLAMPwoFE(DrugLAMPBase):
    def __init__(self, n_drug_feature, n_prot_feature, n_hidden=128, **cfg):
        super().__init__(n_drug_feature, n_prot_feature, n_hidden, **cfg)

    def forward(self, vd, vp, xd, xp, mode="train"):
        fill_bit_p = torch.zeros_like(xp.sum(dim=-1))
        mask_p = xp.sum(dim=-1) == 0
        fill_bit_p[mask_p] = 1
        xp = torch.cat((xp, fill_bit_p.unsqueeze(-1)), dim=-1)

        fill_bit_d = torch.zeros_like(xd.sum(dim=-1))
        mask_d = xd.sum(dim=-1) == 0
        fill_bit_d[mask_d] = 1
        xd = torch.cat((xd, fill_bit_d.unsqueeze(-1)), dim=-1)

        vp_ssl = vp # use to do MLM for xp
        xp_ssl = xp
        fill_bit_p_ssl = fill_bit_p
        vd_ssl = None
        xd_ssl = xd
        ssl = {
            'vp': vp_ssl,
            'xp': xp_ssl,
            'fill_bit_p': fill_bit_p_ssl,
            'vd': vd_ssl,
            'xd': xd_ssl,
            'p_mode': 'xp',
        }

        site_seq_len = self.seq_len_q // self.site_len

        xp = xp.view(-1, self.site_len, site_seq_len, xp.size()[-1])
        xp = torch.mean(xp, dim=1)

        # Encode prots llm
        hx = xp
        xp = self.p_adaptor_wo_skip_connect(xp)
        xp = xp + hx
        xp = self.act_p(self.lin_p1(xp))
        xp = self.p_norm(xp)
        xp = self.lin_p2(xp)

        xd = self.act_d(self.lin_d1(xd))
        xd = self.d_norm(xd)
        xd = self.lin_d2(xd)

        # PGD GCA
        mx, self.A_x_gca = self.x_gca(xp.permute(1, 0, 2), xd.permute(1, 0, 2), xd.permute(1, 0, 2))
        mx = mx.permute(1, 0, 2)

        mx = torch.cat((xp, mx), 2)

        hx = mx
        mx = self.x_mhla(hx)
        mx = mx + hx

        mx = self.x_gca_norm(mx)

        f, self.attn, self.guide_attn = self.pmma(mx, mx)
        f = torch.mean(f, dim=1)
        score = self.mlp_classifier(f) # (B, 256) -> (B, 1)
        if mode == "train":
            return None, None, ssl, None, score
        elif mode == "eval":
            return None, None, score, self.attn