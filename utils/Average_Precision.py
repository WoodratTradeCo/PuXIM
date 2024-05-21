import torch
import numpy as np
from torchmetrics.functional import retrieval_average_precision
import torch.nn.functional as F
from tqdm import tqdm

def calculate_mAP(sk_outputs, im_outputs):
    Len_sk = len(sk_outputs)  # 397
    Len_im = len(im_outputs)
    query_feat_all = torch.cat([sk_outputs[i][0] for i in range(Len_sk)], dim=0)  # 12696
    gallery_feat_all = torch.cat([im_outputs[i][0] for i in range(Len_im)], dim=0)  # 12533
    all_category = np.array(sum([list(sk_outputs[i][1]) for i in range(Len_sk)], []))  # 12696
    all_category_gallery = np.array(sum([list(im_outputs[i][1]) for i in range(Len_im)], []))
    # mAP category-level SBIR Metrics
    gallery = gallery_feat_all
    ap = torch.zeros(len(query_feat_all))
    ap_200 = torch.zeros(len(query_feat_all))
    for idx, sk_feat in enumerate(tqdm(query_feat_all)):
        category = all_category[idx]
        distance = -1 * (1 - F.cosine_similarity(sk_feat.unsqueeze(0), gallery))  # 12533
        target = torch.zeros(len(gallery), dtype=torch.bool)  # 12533
        target[np.where(all_category_gallery == category)] = True
        ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
        ap_200[idx] = retrieval_average_precision(distance.cpu(), target.cpu(), top_k=200)
    mAP = torch.mean(ap)
    mAP_200 = torch.mean(ap_200)
    return mAP.item(), mAP_200.item()


# def calculate_mAP(val_step_outputs):
#     Len = len(val_step_outputs)  # 397
#     query_feat_all = torch.cat([val_step_outputs[i][0] for i in range(Len)], dim=0)
#     gallery_feat_all = torch.cat([val_step_outputs[i][1] for i in range(Len)], dim=0)
#     all_category = np.array(sum([list(val_step_outputs[i][2]) for i in range(Len)], []))
#
#     # mAP category-level SBIR Metrics
#     gallery = gallery_feat_all
#     ap = torch.zeros(len(query_feat_all))
#     for idx, sk_feat in enumerate(tqdm(query_feat_all)):
#         category = all_category[idx]
#         distance = -1 * (1 - F.cosine_similarity(sk_feat.unsqueeze(0), gallery))
#         target = torch.zeros(len(gallery), dtype=torch.bool)
#         target[np.where(all_category == category)] = True
#         ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
#
#     mAP = torch.mean(ap)
#
#     return mAP.item()
