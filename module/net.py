import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from module.vit import Self_Attention
import vxl_sampler
from vxl_sampler import clip
from module.decoder import TransformerDecoder


def get_att_mask(attention, ratio=0.7):
    bs = attention.shape[0]
    mask = torch.ones((bs, 196)).cuda()
    N = int(attention.shape[1]*ratio)
    reservation = torch.argsort(attention, descending=False)
    reservation = reservation[:, :N+1]
    mask = mask.scatter_(1, reservation, 0).cuda()
    return mask


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        '''CLIP VXL Sampler'''
        self.vxl_sampler, _ = vxl_sampler.load("CS-ViT-B/16")
        self.vit = Self_Attention(cls_number=1000, pretrained=True)
        self.mlp_head = nn.Linear(768, 104)
        self.cos_similarity = nn.CosineSimilarity(dim=1)
        self.proj = nn.Linear(512, 768)
        '''second semantic space'''
        self.decoder = TransformerDecoder()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, sk, im, im_text, stage='train'):
        b = sk.size(0) // 2
        if stage == 'train':
            with torch.no_grad():
                im_feat = self.vxl_sampler.encode_image(im)
                im_feat = im_feat / im_feat.norm(dim=1, keepdim=True)
                batch_similarity = []
                for i in range(im_feat.size(0)):
                    txt = ['a', im_text[i]]
                    text_feat = clip.tokenize(txt).to("cuda")
                    text_feat = self.vxl_sampler.encode_text(text_feat.to("cuda"))
                    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                    sim = vxl_sampler.clip_feature_surgery(im_feat[i].unsqueeze(0), text_feat)
                    batch_similarity.append(sim)
                similarity = torch.cat(batch_similarity, dim=0)
                similarity = similarity[:, 1:, -1]  # b*196
                mask = get_att_mask(similarity)

            sk_feat, _, _ = self.vit(sk)
            im_feat, _, _ = self.vit(im)
            if opts.match == "mask":
                im_feat_msk = im_feat[:, 1:] * mask.unsqueeze(-1)
            else:
                im_feat_msk = im_feat
            feat_map = torch.cat((sk_feat, im_feat), 0)
            loss_rec = self.mse(feat_map[feat_map.size(0)//2:, 1:], im_feat_msk)
            feat = feat_map[:, 0]

            sk_int, p_int, n_int = feat_map[:b], feat_map[2 * b:3 * b], feat_map[3 * b:]
            p_int = self.decoder(sk_int, p_int)
            n_int = self.decoder(sk_int, n_int)
            int_feat = torch.cat((p_int, n_int), 0)
            return feat, loss_rec, int_feat

        else:
            feat, _, _ = self.vit(sk)
            feat = feat[:, 0]
            return feat

