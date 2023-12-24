import torch
from .basic_model import DrugLAMPBase

class DrugLAMP(DrugLAMPBase):
    def __init__(self, n_drug_feature, n_prot_feature, n_hidden=128, **cfg):
        super().__init__(n_drug_feature, n_prot_feature, n_hidden, **cfg)

    def forward(self, vd, vp, xd, xp, mode="train"):
        vd = self.drug_extractor(vd) # (64, 512, 128)

        fill_bit_p = torch.zeros_like(xp.sum(dim=-1))
        mask_p = xp.sum(dim=-1) == 0
        fill_bit_p[mask_p] = 1
        xp = torch.cat((xp, fill_bit_p.unsqueeze(-1)), dim=-1)

        fill_bit_d = torch.zeros_like(xd.sum(dim=-1))
        mask_d = xd.sum(dim=-1) == 0
        fill_bit_d[mask_d] = 1
        xd = torch.cat((xd, fill_bit_d.unsqueeze(-1)), dim=-1)

        vp_ssl = vp
        xp_ssl = xp
        fill_bit_p_ssl = fill_bit_p
        vd_ssl = vd
        xd_ssl = xd
        ssl = {
            'vp': vp_ssl,
            'xp': xp_ssl,
            'fill_bit_p': fill_bit_p_ssl,
            'vd': vd_ssl,
            'xd': xd_ssl,
        }

        vp = self.protein_extractor(vp, fill_bit_p)
        site_seq_len = self.seq_len_q // self.site_len
        vp = vp.view(-1, self.site_len, site_seq_len, vp.size()[-1])
        vp = torch.mean(vp, dim=1)

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
        mv, self.A_v_gca = self.v_gca(vp.permute(1, 0, 2), vd.permute(1, 0, 2), vd.permute(1, 0, 2))
        mx, self.A_x_gca = self.x_gca(xp.permute(1, 0, 2), xd.permute(1, 0, 2), xd.permute(1, 0, 2))
        mv = mv.permute(1, 0, 2)
        mx = mx.permute(1, 0, 2)

        mv = torch.cat((vp, mv), 2)
        mx = torch.cat((xp, mx), 2)

        hv = mv
        hx = mx
        mv = self.v_mhla(hv)
        mx = self.x_mhla(hx)
        mv = mv + hv
        mx = mx + hx

        mv = self.v_gca_norm(mv)
        mx = self.x_gca_norm(mx)

        f, self.attn, self.guide_attn = self.pmma(mx, mv)
        f = torch.mean(f, dim=1)
        score = self.mlp_classifier(f) # (B, 256) -> (B, 1)
        if mode == "train":
            return vd, vp, ssl, None, score
        elif mode == "eval":
            return vd, vp, score, self.attn