import os
import datetime
from utils.options import opts
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu
from torch.utils.data import DataLoader
from Data.Sketchy import Sketchy
from Data.dataset import load_data
from utils.Average_Precision import calculate_mAP
# from utils.map import calculate_mAP
from utils.AverageMeter import AverageMeter
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import Logger
import torch
import logging
from module.net import Model
import timm.scheduler
import numpy as np


logging.getLogger('PIL').setLevel(logging.WARNING)
torch.autograd.set_detect_anomaly(True)
dataset_transforms = Sketchy.data_transform(opts)
train_data, sk_valid_data, im_valid_data = load_data(opts)

train_loader = DataLoader(dataset=train_data, batch_size=opts.batch_size, num_workers=opts.workers)
sk_valid_loader = DataLoader(dataset=sk_valid_data, batch_size=8, num_workers=opts.workers)
im_valid_loader = DataLoader(dataset=im_valid_data, batch_size=8, num_workers=opts.workers)

exp_dir = os.path.join(opts.exp_base, opts.exp)
exp_log_dir = os.path.join(exp_dir, "log")
if not os.path.exists(exp_log_dir):
    os.makedirs(exp_log_dir)

exp_ckpt_dir = os.path.join(exp_dir, "results/checkpoints")
if not os.path.exists(exp_ckpt_dir):
    os.makedirs(exp_ckpt_dir)

now_str = datetime.datetime.now().__str__().replace(' ', '_').replace(':', '.')

logger_path = os.path.join(exp_log_dir, opts.exp + ".log")
logger = Logger(logger_path).get_logger()

net = Model().cuda().float()
# pretrained_dict = torch.load("./results/sketchy25/full/checkpoints/best.pth")["network"]
# net.load_state_dict(pretrained_dict, strict=False)
if opts.match == 'mask':
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
else:
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
                                                       margin=0.2)

bce_loss_fn = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-3)


def train_function(epoch_idx):
    training_loss = AverageMeter()
    display_step = 30
    net.train()

    for idx, (sk_tensor, img_tensor, sk_neg, neg_tensor, sk_label, im_label, sk_neg_label, neg_label, im_text, im_n_text
              ) in enumerate(tqdm(train_loader)):
        sk_tensor, img_tensor, sk_neg, neg_tensor = sk_tensor.cuda(), img_tensor.cuda(), sk_neg.cuda(), neg_tensor.cuda()
        Len = sk_tensor.size(0)

        sk = torch.cat((sk_tensor, sk_neg), dim=0)
        im = torch.cat((img_tensor, neg_tensor), dim=0)
        text = im_text + im_n_text
        # model
        feat, loss_rec, int_feat = net(sk, im, text, stage='train')
        sk = feat[0:Len]
        im_p = feat[2 * Len:3 * Len]
        im_n = feat[3 * Len:]

        int_label = torch.cat((torch.ones(Len).cuda(), torch.zeros(Len).cuda()), dim=0)

        batch_loss_tri = triplet_loss_fn(sk, im_p, im_n)
        batch_loss_rec = loss_rec
        batch_loss_int = bce_loss_fn(int_feat, int_label.long())

        batch_loss = batch_loss_tri + 5e-2 * batch_loss_rec + 0.1 * batch_loss_int
        training_loss.update(batch_loss.item(), sk.size(0))
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if (idx + 1) % display_step == 0:
            logger.info(
                "==> Iteration [{}][{}/{}]:".format(epoch_idx + 1, idx + 1, len(train_loader)))
            logger.info("current batch loss: {}  triplet_loss: {}  mse: {}  itm: {}".format(
                batch_loss.item(), batch_loss_tri.item(), 5e-2 * batch_loss_rec.item(), 0.1 * batch_loss_int.item()))
            logger.info("average loss: {}".format(
                training_loss.avg))
    out_sk, out_im = validate_function(sk_valid_loader, im_valid_loader)
    mAP, mAP_200, prec_200, prec_100 = calculate_mAP(out_sk, out_im)
    return mAP, mAP_200, prec_200, prec_100


def validate_function(sk_loader, im_loader):
    sk_val_outputs = []
    im_val_outputs = []
    net.eval()
    # TODO
    with torch.no_grad():
        logger.info("loading photo valid......")
        for i, (im, im_label, im_category) in enumerate(tqdm(im_loader)):

            im = im.cuda()
            im_feat = net(im, None, None, stage='val')
            im_val_outputs.append([im_feat.cpu().numpy(), im_label.cpu().numpy()])

        logger.info("loading sketch valid......")
        for i, (sk, sk_label, sk_category) in enumerate(tqdm(sk_loader)):

            sk = sk.cuda()
            sk_feat = net(sk, None, None, stage='val')
            sk_val_outputs.append([sk_feat.cpu().numpy(), sk_label.cpu().numpy()])
    return sk_val_outputs, im_val_outputs


if __name__ == '__main__':
    logger.info("optimizer: {} epoch: {}".format(optimizer, opts.epoch))
    best_metric, best_metric_200, best_prec_200, max_val_epoch = 1e-3, 1e-3, 1e-3, 0
    alpha = 0
    for e in range(opts.epoch):
        mAP, mAP_200, prec_200, prec_100 = train_function(e)
        logger.info("Begin Evaluating")
        if best_metric > mAP:
            best_metric = best_metric
        else:
            best_metric = mAP
            best_metric_200 = mAP_200
            best_prec_200 = prec_200
            best_prec_100 = prec_100
            max_val_epoch = e + 1
            net_checkpoint_name = "best.pth"  # opts.exp + "_net_epoch" + str(max_val_epoch)
            net_checkpoint_path = os.path.join(exp_ckpt_dir, net_checkpoint_name)
            net_state = {"epoch": e + 1, "network": net.state_dict()}
            torch.save(net_state, net_checkpoint_path)

        logger.info("mAP: {}  mAP@200: {}  prec@200: {}  prec@100: {}".format(mAP, mAP_200, prec_200, prec_100))
        logger.info("Best mAP: {}  Best mAP@200: {}  Best prec@100: {}  max_epoch: {}".format(
            best_metric, best_metric_200, best_prec_200, max_val_epoch))