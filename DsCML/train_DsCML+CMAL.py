#!/usr/bin/env python
import sys
import os
sys.path.append(os.getcwd())
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from DsCML.common.solver.build import build_optimizer, build_scheduler
from DsCML.common.utils.checkpoint import CheckpointerV2
from DsCML.common.utils.logger import setup_logger
from DsCML.common.utils.metric_logger import MetricLogger
from DsCML.common.utils.torch_util import set_random_seed
from DsCML.models.build import build_model_2d, build_model_3d, build_2d_L2G, build_3d_L2G, build_S2DT3D_Dis, build_S3DT2D_Dis
from DsCML.data.build import build_dataloader
from DsCML.data.utils.validate import validate
from DsCML.models.losses import entropy_loss
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='DsCML training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )

    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args

def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate_D(cfg, optimizer, i_iter):
    lr = lr_poly(cfg.OPTIMIZER_D1.BASE_LR, i_iter, cfg.SCHEDULER.MAX_ITERATION, cfg.OPTIMIZER_D1.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def train(cfg, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    logger = logging.getLogger('DsCML.train')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d, train_metric_2d = build_model_2d(cfg)
    logger.info('Build 2D model:\n{}'.format(str(model_2d)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    # build cross modal learning head for 2D
    LH_2d = build_2d_L2G(cfg)
    logger.info('Build 2D model learning head:\n{}'.format(str(LH_2d)))
    num_params = sum(param.numel() for param in LH_2d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    # build 3d model
    model_3d, train_metric_3d = build_model_3d(cfg)
    logger.info('Build 3D model:\n{}'.format(str(model_3d)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    # build cross modal learning head for 3D
    LH_3d = build_3d_L2G(cfg)
    logger.info('Build 3D model learning head:\n{}'.format(str(LH_3d)))
    num_params = sum(param.numel() for param in LH_3d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    # build Discriminator for S_2D to T_3D
    model_D1 = build_S2DT3D_Dis(cfg)
    logger.info('Build Discriminator for S_2D to T_3D')

    # build Discriminator for S_3D to T_2D
    model_D2 = build_S3DT2D_Dis(cfg)
    logger.info('Build Discriminator for S_2D to T_3D')




    model_2d = model_2d.cuda()
    LH_2d = LH_2d.cuda()
    model_3d = model_3d.cuda()
    LH_3d = LH_3d.cuda()
    model_D1 = model_D1.cuda()
    model_D2 = model_D2.cuda()

    # build optimizer
    optimizer_2d = build_optimizer(cfg, model_2d,LH_2d)
    optimizer_3d = build_optimizer(cfg, model_3d,LH_3d)

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=cfg.OPTIMIZER_D1.BASE_LR, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=cfg.OPTIMIZER_D2.BASE_LR, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    # build lr scheduler
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_2d = CheckpointerV2(model_2d,
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d = checkpointer_2d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)

    checkpointer_2d_LH = CheckpointerV2(LH_2d,
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d_LH',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d_LH = checkpointer_2d_LH.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)

    checkpointer_3d = CheckpointerV2(model_3d,
                                     optimizer=optimizer_3d,
                                     scheduler=scheduler_3d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d = checkpointer_3d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)

    checkpointer_3d_LH = CheckpointerV2(LH_3d,
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d_LH',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d_LH = checkpointer_3d_LH.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)

    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_2d.get('iteration', 0)

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration)
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target') if val_period > 0 else None

    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None)
    }
    best_metric_iter = {'2d': -1, '3d': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d])
    val_metric_logger = MetricLogger(delimiter='  ')

    def setup_train():
        # set training mode
        model_2d.train()
        LH_2d.train()
        model_3d.train()
        LH_3d.train()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_2d.eval()
        LH_2d.eval()
        model_3d.eval()
        LH_3d.eval()
        # reset metric
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    setup_train()
    end = time.time()
    train_iter_src = enumerate(train_dataloader_src)
    train_iter_trg = enumerate(train_dataloader_trg)

    l1_loss_fn = torch.nn.L1Loss(reduce=True,size_average=True)

    def cc(a,b):

        a = a.view(-1)
        b = b.view(-1)
        g_s_m = pd.Series(a.cpu().detach().numpy())
        g_a_d = pd.Series(b.cpu().detach().numpy())
        corr_gust = round(g_s_m.corr(g_a_d), 4)
        return torch.tensor(cfg.TRAIN.XMUDA.lambda_cc * (1 - corr_gust) / 2).cuda()


    for iteration in range(start_iteration, max_iteration):
        # fetch data_batches for source & target
        _, data_batch_src = train_iter_src.__next__()
        _, data_batch_trg = train_iter_trg.__next__()
        data_time = time.time() - end
        # copy data from cpu to gpu
        if 'SCN' in cfg.DATASET_SOURCE.TYPE and 'SCN' in cfg.DATASET_TARGET.TYPE:
            # source
            data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
            data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
            data_batch_src['img'] = data_batch_src['img'].cuda()
            # target
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
            data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch_trg['img'] = data_batch_trg['img'].cuda()
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                data_batch_trg['pseudo_label_2d'] = data_batch_trg['pseudo_label_2d'].cuda()
                data_batch_trg['pseudo_label_3d'] = data_batch_trg['pseudo_label_3d'].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()

        #---------------------------------------------------

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(cfg, optimizer=optimizer_D1, i_iter=iteration)
        adjust_learning_rate_D(cfg, optimizer=optimizer_D2, i_iter=iteration)

        source_label = 0
        target_label = 1
        # ---------------------------------------------------



        # train G **********************************************************************************
        for param in model_D1.parameters():
            param.requires_grad = False

        for param in model_D2.parameters():
            param.requires_grad = False

        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #

        preds_2d_fe, out_2D_feature, img_indices = model_2d(data_batch_src)
        preds_2d_be = LH_2d(out_2D_feature, img_indices)
        preds_3d_fe, out_3D_feature = model_3d(data_batch_src)
        preds_3d_be = LH_3d(out_3D_feature)

        # segmentation loss: cross entropy
        seg_loss_src_2d = F.cross_entropy(preds_2d_fe['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        seg_loss_src_3d = F.cross_entropy(preds_3d_fe['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        train_metric_logger.update(seg_loss_src_2d=seg_loss_src_2d, seg_loss_src_3d=seg_loss_src_3d)
        loss_2d = seg_loss_src_2d
        loss_3d = seg_loss_src_3d
        loss_global = 0

        if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
            # cross-modal loss: KL divergence

            seg_logit_2d_avg = preds_2d_be['seg_logit_avg']
            seg_logit_2d_max = preds_2d_be['seg_logit_max']
            seg_logit_2d_min = preds_2d_be['seg_logit_min']
            seg_logit_3d_point = preds_3d_be['seg_logit_point']
            seg_logit_2d_global = preds_2d_be['seg_logit_global']
            seg_logit_3d_global = preds_3d_be['seg_logit_global']

            src_2d_adv_h1 = preds_2d_fe['seg_logit']
            src_2d_adv_h2 = preds_2d_be['seg_logit_avg']
            src_3d_adv_h1 = preds_3d_fe['seg_logit']
            src_3d_adv_h2 = preds_3d_be['seg_logit_point']





            xm_loss_src_2d_avg = F.kl_div(F.log_softmax(seg_logit_2d_avg, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean() + cc(F.log_softmax(seg_logit_2d_avg, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1))



            xm_loss_src_2d_max = F.kl_div(F.log_softmax(seg_logit_2d_max, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean() + cc(F.log_softmax(seg_logit_2d_max, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1))



            xm_loss_src_2d_min = F.kl_div(F.log_softmax(seg_logit_2d_min, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean() + cc(F.log_softmax(seg_logit_2d_min, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1))



            xm_loss_src_2d = (xm_loss_src_2d_max + xm_loss_src_2d_min)/2
            xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d_point, dim=1),
                                      F.softmax(preds_2d_fe['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean() + cc(F.log_softmax(seg_logit_3d_point, dim=1),
                                      F.softmax(preds_2d_fe['seg_logit'].detach(), dim=1))



            train_metric_logger.update(xm_loss_src_2d=xm_loss_src_2d,
                                       xm_loss_src_3d=xm_loss_src_3d)
            loss_2d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d
            loss_3d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d
            loss_global = cfg.TRAIN.XMUDA.lambda_xm_global_src * l1_loss_fn(seg_logit_2d_global,seg_logit_3d_global)


        # update metric (e.g. IoU)
        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d_fe, data_batch_src)
            train_metric_3d.update_dict(preds_3d_fe, data_batch_src)

        # backward
        loss_2d.backward(retain_graph=True)
        loss_3d.backward(retain_graph=True)
        loss_global.backward()

        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #

        preds_2d_fe, out_2D_feature, img_indices = model_2d(data_batch_trg)
        preds_2d_be = LH_2d(out_2D_feature, img_indices)
        preds_3d_fe, out_3D_feature = model_3d(data_batch_trg)
        preds_3d_be = LH_3d(out_3D_feature)

        loss_2d = []
        loss_3d = []
        loss_global = 0
        if cfg.TRAIN.XMUDA.lambda_xm_trg > 0:
            # cross-modal loss: KL divergence

            seg_logit_2d_avg = preds_2d_be['seg_logit_avg']
            seg_logit_2d_max = preds_2d_be['seg_logit_max']
            seg_logit_2d_min = preds_2d_be['seg_logit_min']
            seg_logit_3d_point = preds_3d_be['seg_logit_point']
            seg_logit_2d_global = preds_2d_be['seg_logit_global']
            seg_logit_3d_global = preds_3d_be['seg_logit_global']

            trg_2d_adv_h1 = preds_2d_fe['seg_logit']
            trg_2d_adv_h2 = preds_2d_be['seg_logit_avg']
            trg_3d_adv_h1 = preds_3d_fe['seg_logit']
            trg_3d_adv_h2 = preds_3d_be['seg_logit_point']




            xm_loss_trg_2d_avg = F.kl_div(F.log_softmax(seg_logit_2d_avg, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean() + cc(F.log_softmax(seg_logit_2d_avg, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1))



            xm_loss_trg_2d_max = F.kl_div(F.log_softmax(seg_logit_2d_max, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean() + cc(F.log_softmax(seg_logit_2d_max, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1))


            xm_loss_trg_2d_min = F.kl_div(F.log_softmax(seg_logit_2d_min, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean() + cc(F.log_softmax(seg_logit_2d_min, dim=1),
                                      F.softmax(preds_3d_fe['seg_logit'].detach(), dim=1))


            xm_loss_trg_2d = (xm_loss_trg_2d_max + xm_loss_trg_2d_min)/2
            xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d_point, dim=1),
                                      F.softmax(preds_2d_fe['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean() + cc(F.log_softmax(seg_logit_3d_point, dim=1),
                                      F.softmax(preds_2d_fe['seg_logit'].detach(), dim=1))



            train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                       xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_3d)

            loss_global = cfg.TRAIN.XMUDA.lambda_xm_global_trg * l1_loss_fn(seg_logit_2d_global, seg_logit_3d_global)

        if cfg.TRAIN.XMUDA.lambda_pl > 0:
            # uni-modal self-training loss with pseudo labels
            pl_loss_trg_2d = F.cross_entropy(preds_2d_fe['seg_logit'], data_batch_trg['pseudo_label_2d'])
            pl_loss_trg_3d = F.cross_entropy(preds_3d_fe['seg_logit'], data_batch_trg['pseudo_label_3d'])
            train_metric_logger.update(pl_loss_trg_2d=pl_loss_trg_2d,
                                       pl_loss_trg_3d=pl_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_3d)
        if cfg.TRAIN.XMUDA.lambda_minent > 0:

            # MinEnt
            minent_loss_trg_2d = entropy_loss(F.softmax(preds_2d_fe['seg_logit'], dim=1))
            minent_loss_trg_3d = entropy_loss(F.softmax(preds_3d_fe['seg_logit'], dim=1))
            train_metric_logger.update(minent_loss_trg_2d=minent_loss_trg_2d,
                                       minent_loss_trg_3d=minent_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_3d)

        # adv target train

        # D_target_2D_head1 = model_D2([F.softmax(trg_2d_adv_h1,dim=1), data_batch_trg['x'][1]])['Dis_out_point']
        # D_target_2D_head2 = model_D2([F.softmax(trg_2d_adv_h2,dim=1), data_batch_trg['x'][1]])['Dis_out_point']
        # D_target_3D_head1 = model_D1([F.softmax(trg_3d_adv_h1,dim=1), data_batch_trg['x'][1]])['Dis_out_point']
        # D_target_3D_head2 = model_D1([F.softmax(trg_3d_adv_h2,dim=1), data_batch_trg['x'][1]])['Dis_out_point']
        D_target_2D_head1 = model_D2(F.softmax(trg_2d_adv_h1,dim=1))['Dis_out_batch']
        D_target_2D_head2 = model_D2(F.softmax(trg_2d_adv_h2,dim=1))['Dis_out_batch']
        D_target_3D_head1 = model_D1(F.softmax(trg_3d_adv_h1,dim=1))['Dis_out_batch']
        D_target_3D_head2 = model_D1(F.softmax(trg_3d_adv_h2,dim=1))['Dis_out_batch']


        loss_adv_target2D = cfg.TRAIN.XMUDA.G_adv_trg_2d * ( bce_loss(D_target_2D_head1, Variable(torch.FloatTensor(D_target_2D_head1.data.size()).fill_(source_label),requires_grad=True).cuda()) + bce_loss(D_target_2D_head2, Variable(torch.FloatTensor(D_target_2D_head2.data.size()).fill_(source_label),requires_grad=True).cuda()) )
        loss_adv_target3D = cfg.TRAIN.XMUDA.G_adv_trg_3d * ( bce_loss(D_target_3D_head1, Variable(torch.FloatTensor(D_target_3D_head1.data.size()).fill_(source_label),requires_grad=True).cuda()) + bce_loss(D_target_3D_head2, Variable(torch.FloatTensor(D_target_3D_head2.data.size()).fill_(source_label),requires_grad=True).cuda()) )
        loss_adv_trg = loss_adv_target2D + loss_adv_target3D

        train_metric_logger.update(G_adv_trg_2d=loss_adv_target2D,
                                   G_adv_trg_3d=loss_adv_target3D)


        loss_adv_trg.backward(retain_graph=True)
        sum(loss_2d).backward(retain_graph=True)
        sum(loss_3d).backward(retain_graph=True)
        loss_global.backward()

        optimizer_2d.step()
        optimizer_3d.step()

        # train D **********************************************************************************

        # bring back requires_grad
        for param in model_D1.parameters():
            param.requires_grad = True

        for param in model_D2.parameters():
            param.requires_grad = True

        # adv source train

        src_2d_adv_h1 = src_2d_adv_h1.detach()
        src_2d_adv_h2 = src_2d_adv_h2.detach()
        src_3d_adv_h1 = src_3d_adv_h1.detach()
        src_3d_adv_h2 = src_3d_adv_h2.detach()

        D_source_2D_head1 = model_D1(F.softmax(src_2d_adv_h1,dim=1))['Dis_out_batch']
        D_source_2D_head2 = model_D1(F.softmax(src_2d_adv_h2,dim=1))['Dis_out_batch']
        D_source_3D_head1 = model_D2(F.softmax(src_3d_adv_h1,dim=1))['Dis_out_batch']
        D_source_3D_head2 = model_D2(F.softmax(src_3d_adv_h2,dim=1))['Dis_out_batch']

        loss_adv_source2D = cfg.TRAIN.XMUDA.D_adv_src_2d * ( bce_loss(D_source_2D_head1, Variable(torch.FloatTensor(D_source_2D_head1.data.size()).fill_(source_label),requires_grad=True).cuda()) + bce_loss(D_source_2D_head2, Variable(torch.FloatTensor(D_source_2D_head2.data.size()).fill_(source_label),requires_grad=True).cuda()) )
        loss_adv_source3D = cfg.TRAIN.XMUDA.D_adv_src_3d * ( bce_loss(D_source_3D_head1, Variable(torch.FloatTensor(D_source_3D_head1.data.size()).fill_(source_label),requires_grad=True).cuda()) + bce_loss(D_source_3D_head2, Variable(torch.FloatTensor(D_source_3D_head2.data.size()).fill_(source_label),requires_grad=True).cuda()) )
        loss_adv_src = loss_adv_source2D + loss_adv_source3D

        train_metric_logger.update(D_adv_src_2d=loss_adv_source2D,
                                   D_adv_src_3d=loss_adv_source3D)

        loss_adv_src.backward(retain_graph=True)

        # adv target train

        trg_2d_adv_h1 = trg_2d_adv_h1.detach()
        trg_2d_adv_h2 = trg_2d_adv_h2.detach()
        trg_3d_adv_h1 = trg_3d_adv_h1.detach()
        trg_3d_adv_h2 = trg_3d_adv_h2.detach()

        D_target_2D_head1 = model_D2(F.softmax(trg_2d_adv_h1,dim=1))['Dis_out_batch']
        D_target_2D_head2 = model_D2(F.softmax(trg_2d_adv_h2,dim=1))['Dis_out_batch']
        D_target_3D_head1 = model_D1(F.softmax(trg_3d_adv_h1,dim=1))['Dis_out_batch']
        D_target_3D_head2 = model_D1(F.softmax(trg_3d_adv_h2,dim=1))['Dis_out_batch']

        loss_adv_target2D = cfg.TRAIN.XMUDA.D_adv_trg_2d * ( bce_loss(D_target_2D_head1, Variable(
            torch.FloatTensor(D_target_2D_head1.data.size()).fill_(target_label)).cuda()) + bce_loss(D_target_2D_head2,Variable(torch.FloatTensor(D_target_2D_head2.data.size()).fill_(target_label)).cuda()) )
        loss_adv_target3D = cfg.TRAIN.XMUDA.D_adv_trg_3d * ( bce_loss(D_target_3D_head1, Variable(
            torch.FloatTensor(D_target_3D_head1.data.size()).fill_(target_label)).cuda()) + bce_loss(D_target_3D_head2,Variable(torch.FloatTensor(D_target_3D_head2.data.size()).fill_(target_label)).cuda()) )
        loss_adv_trg = loss_adv_target2D + loss_adv_target3D

        train_metric_logger.update(D_adv_trg_2d=loss_adv_target2D,
                                   D_adv_trg_3d=loss_adv_target3D)

        loss_adv_trg.backward()

        optimizer_D1.step()
        optimizer_D2.step()

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)

        train_metric_logger.update(dis_optim_lr=optimizer_D1.param_groups[0]['lr'])

        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr=optimizer_2d.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
            checkpointer_2d_LH.save('model_2d_LH{:06d}'.format(cur_iter), **checkpoint_data_2d_LH)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)
            checkpointer_3d_LH.save('model_3d_LH{:06d}'.format(cur_iter), **checkpoint_data_3d_LH)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()

            validate(cfg,
                     model_2d,
                     model_3d,
                     val_dataloader,
                     val_metric_logger)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in ['2d', '3d']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter

            # restore training
            setup_train()

        scheduler_2d.step()
        scheduler_3d.step()
        end = time.time()

    for modality in ['2d', '3d']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(modality.upper(),
                                                                     cfg.VAL.METRIC,
                                                                     best_metric[modality] * 100,
                                                                     best_metric_iter[modality]))


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from DsCML.common.config import purge_cfg
    from DsCML.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('DsCML', output_dir, comment='train.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    assert cfg.TRAIN.XMUDA.lambda_xm_src > 0 or cfg.TRAIN.XMUDA.lambda_xm_trg > 0 or cfg.TRAIN.XMUDA.lambda_pl > 0 or \
           cfg.TRAIN.XMUDA.lambda_minent > 0
    train(cfg, output_dir, run_name)


if __name__ == '__main__':
    main()
