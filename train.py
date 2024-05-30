#!/usr/bin/python
# -*- encoding: utf-8 -*-
from cgi import print_arguments
from PIL import Image
import numpy as np
from torch.autograd import Variable
from new_fusenet import FusionNet

from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from model_TII import BiSeNet
from cityscapes import CityScapes
from loss import OhemCELoss, Fusionloss, NewFusionloss
from optimizer import Optimizer
import torch

torch.manual_seed(3407)

from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = '4' 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

fuse_batch_size = 5   
test_batch_size = 6
seg_batch_size = 6  

fuse_num_workers = 4
test_num_workers = 4
seg_num_workers = 2  


def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3) 
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def train_seg(i=0, logger=None):

    load_path = './model/model_final.pth'         
    modelpth = './model'                                 
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    os.makedirs(modelpth, mode=0o777, exist_ok=True)

    n_classes = 9                    
    n_img_per_gpu = seg_batch_size   

    n_workers = 4                   
    cropsize = [640, 480]            

    ds = CityScapes('./MSRS/', cropsize=cropsize, mode='train',
                    Method=Method) 
    dl = DataLoader(
        ds,
        batch_size=n_img_per_gpu,
        shuffle=False,
        num_workers=seg_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    ignore_idx = 255
    net = BiSeNet(n_classes=n_classes)  

    if i > 0:                            
        net.load_state_dict(torch.load(load_path, map_location='cuda:4'))

    net.cuda()
    net.train()   

    print('Load Pre-trained Segmentation Model:{}!'.format(load_path)) 
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16

    criteria_p = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    it_start = i*20000
    iter_nums = 20000

    optim = Optimizer(
        model=net,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
        it=it_start,
    )

    msg_iter = 200
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)   
    epoch = 0
    for it in range(iter_nums):
        try:
            im, lb, _ = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            diter = iter(dl)
            im, lb, _ = next(diter)
        im = im.cuda()   
        lb = lb.cuda()   
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, mid = net(im)  
        lossp = criteria_p(out, lb)
        loss2 = criteria_16(mid, lb)
        loss = lossp + 0.75 * loss2
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())  
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg) 

            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            msg = ', '.join(
                [
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                    'date: {date}',
                ]
            ).format(
                it=it_start+it + 1, max_it=max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta, date=date
            )
            logger.info(msg)
            loss_avg = []
            st = ed

    save_pth = osp.join(modelpth, 'model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()

    torch.save(state, save_pth)
    logger.info(
        'Segmentation Model Training done~, The Model is saved to: {}'.format(
            save_pth)
    )
    logger.info('\n')

    save_pth_bak = osp.join(modelpth, f'model_final_bak_{i}.pth')
    torch.save(state, save_pth_bak)
    logger.info(
        'Segmentation Model Training done~, The Model is saved to: {}'.format(
            save_pth_bak)
    )
    logger.info('\n')


def train_fusion(num=0, logger=None):

    lr_start = 0.001
    modelpth = './model' 
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)

    fusionmodel = FusionNet()
    fusionmodel.cuda()
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)

    if num > 2:
        n_classes = 9
        segmodel = BiSeNet(n_classes=n_classes)
        save_pth = osp.join(modelpth, 'model_final.pth')
        if logger == None:
            logger = logging.getLogger()
            setup_logger(modelpth)
        segmodel.load_state_dict(torch.load(save_pth,  map_location='cuda:4'))
        segmodel.cuda()
        segmodel.eval()
        for p in segmodel.parameters():
            p.requires_grad = False
        print('Load Segmentation Model {} Sucessfully'.format(save_pth))

    if num > 2:
        train_dataset = Fusion_dataset('train')

        print("the training dataset is length:{}".format(train_dataset.length))
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=fuse_batch_size,
            shuffle=True,
            num_workers=fuse_num_workers,
            pin_memory=True,
            drop_last=True,
        )
        train_loader.n_iter = len(train_loader)
    else:
        train_dataset = Fusion_dataset('val','./new_infrared/MSRS/','./new_visible/MSRS/')

        print("the training dataset is length:{}".format(train_dataset.length))
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=fuse_batch_size,
            shuffle=True,
            num_workers=fuse_num_workers,
            pin_memory=True,
            drop_last=True,
        )
        train_loader.n_iter = len(train_loader)
    
    if num > 2:
        score_thres = 0.7
        ignore_idx = 255
        n_min = 8 * 640 * 480 // 8
        criteria_p = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        criteria_16 = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    epoch = 10
    st = glob_st = time.time()
    logger.info('Training Fusion Model start')

    if num > 2:
        criteria_fusion = Fusionloss()
        for epo in range(0, epoch):
            lr_start = 0.001
            lr_decay = 0.75
            lr_this_epo = lr_start * lr_decay ** (epo - 1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_epo
            for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
                fusionmodel.train()
                image_vis = Variable(image_vis).cuda()
                image_vis_ycrcb = RGB2YCrCb(image_vis)
                image_ir = Variable(image_ir).cuda()
                label = Variable(label).cuda()
                logits = fusionmodel(image_vis_ycrcb, image_ir)
                fusion_ycrcb = torch.cat(
                    (logits, image_vis_ycrcb[:, 1:2, :, :],
                    image_vis_ycrcb[:, 2:, :, :]),
                    dim=1,
                )
                fusion_image = YCrCb2RGB(fusion_ycrcb)

                ones = torch.ones_like(fusion_image)
                zeros = torch.zeros_like(fusion_image)
                fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
                fusion_image = torch.where(
                    fusion_image < zeros, zeros, fusion_image)
                lb = torch.squeeze(label, 1)
                optimizer.zero_grad()

                if num > 2:
                    out, mid = segmodel(fusion_image)
                    lossp = criteria_p(out, lb)
                    loss2 = criteria_16(mid, lb)
                    seg_loss = lossp + 0.1 * loss2

                loss_fusion, loss_in, loss_grad = criteria_fusion(
                    image_vis_ycrcb, image_ir, label, logits, num
                )
                if num > 2:
                    loss_total = loss_fusion + (num) * seg_loss
                else:
                    loss_total = loss_fusion

                loss_total.backward()
                optimizer.step()

                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = train_loader.n_iter * epo + it + 1
                eta = int((train_loader.n_iter * epoch - now_it)
                        * (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                if now_it % 100 == 0:
                    if num > 2:
                        loss_seg = seg_loss.item()
                    else:
                        loss_seg = 0
                    msg = ', '.join(
                        [
                            'step: {it}/{max_it}',
                            'loss_total: {loss_total:.4f}',
                            'loss_in: {loss_in:.4f}',
                            'loss_grad: {loss_grad:.4f}',
                            'loss_seg: {loss_seg:.4f}',
                            'eta: {eta}',
                            'time: {time:.4f}',
                            'date:{date}'
                        ]
                    ).format(
                        it=now_it,
                        max_it=train_loader.n_iter * epoch,
                        loss_total=loss_total.item(),
                        loss_in=loss_in.item(),
                        loss_grad=loss_grad.item(),
                        loss_seg=loss_seg,
                        time=t_intv,
                        eta=eta,
                        date=date,
                    )
                    logger.info(msg)
                    st = ed
    
    else:
        criteria_fusion = NewFusionloss()
        for epo in range(0, epoch):
            lr_start = 0.001
            lr_decay = 0.75
            lr_this_epo = lr_start * lr_decay ** (epo - 1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_epo
            for it, (image_vis, image_ir,  name) in enumerate(train_loader):
                fusionmodel.train()
                image_vis = Variable(image_vis).cuda()
                image_vis_ycrcb = RGB2YCrCb(image_vis)
                image_ir = Variable(image_ir).cuda()
                logits = fusionmodel(image_vis_ycrcb, image_ir)

                fusion_ycrcb = torch.cat(
                    (logits, image_vis_ycrcb[:, 1:2, :, :],
                    image_vis_ycrcb[:, 2:, :, :]),
                    dim=1,
                )
                fusion_image = YCrCb2RGB(fusion_ycrcb)

                ones = torch.ones_like(fusion_image)
                zeros = torch.zeros_like(fusion_image)
                fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
                fusion_image = torch.where(
                    fusion_image < zeros, zeros, fusion_image)
                optimizer.zero_grad()
                label = "default"
                loss_fusion, loss_in, loss_grad = criteria_fusion(
                    image_vis_ycrcb, image_ir, label, logits, num
                )

                if num > 2:
                    loss_total = loss_fusion 
                else:
                    loss_total = loss_fusion

                loss_total.backward()
                optimizer.step()

                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = train_loader.n_iter * epo + it + 1
                eta = int((train_loader.n_iter * epoch - now_it)
                        * (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                if now_it % 100 == 0:
                    if num > 2:
                        loss_seg = seg_loss.item()
                    else:
                        loss_seg = 0
                    msg = ', '.join(
                        [
                            'step: {it}/{max_it}',
                            'loss_total: {loss_total:.4f}',
                            'loss_in: {loss_in:.4f}',
                            'loss_grad: {loss_grad:.4f}',
                            'loss_seg: {loss_seg:.4f}',
                            'eta: {eta}',
                            'time: {time:.4f}',
                            'date:{date}'
                        ]
                    ).format(
                        it=now_it,
                        max_it=train_loader.n_iter * epoch,
                        loss_total=loss_total.item(),
                        loss_in=loss_in.item(),
                        loss_grad=loss_grad.item(),
                        loss_seg=loss_seg,
                        time=t_intv,
                        eta=eta,
                        date=date,
                    )
                    logger.info(msg)
                    st = ed
    
    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)            
    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')

    fusion_model_bak_pth = os.path.join(modelpth, f'fusion_model_bak_{i}.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_bak_pth)            
    logger.info("Fusion Model Save to: {}".format(fusion_model_bak_pth))
    logger.info('\n')

def run_fusion(type='train'):

    fusion_model_path = './model/Fusion/fusion_model.pth'      
    fused_dir = os.path.join('./MSRS/Fusion', type, 'MSRS')   
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = FusionNet()
    fusionmodel.eval()                                        
    fusionmodel.cuda()
    fusionmodel.load_state_dict(torch.load(fusion_model_path, map_location='cuda:4')) 
    print('done!')

    test_dataset = Fusion_dataset(type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=test_num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)               
    with torch.no_grad():
        for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            labels = Variable(labels)
            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
            images_vis_ycrcb = RGB2YCrCb(images_vis)
            logits = fusionmodel(images_vis_ycrcb, images_ir)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :,
                 :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='Fusion')
    parser.add_argument('--batch_size', '-B', type=int,
                        default=fuse_batch_size)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    args = parser.parse_args()

    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)

    start_total = time.time()

    for i in range(4):

        fuse_start = time.time()
        train_fusion(i, logger)
        print("|{0} Train Fusion Model Sucessfully".format(i + 1))
        fuse_end = time.time()
        fuse_time = time.strftime(
            "%H:%M:%S", time.gmtime(fuse_end - fuse_start))
        print("|{0} Train Fusion Model Time: {1} s".format(i + 1, fuse_time))
        logger.info(f'训练模型时间：{fuse_time}\n\n')

        fusion_start = time.time()
        run_fusion('train')
        print("|{0} Fusion Image Sucessfully".format(i + 1))
        fusion_end = time.time()
        fusion_time = time.strftime(
            "%H:%M:%S", time.gmtime(fusion_end - fusion_start))
        print("|{0} Fusion Image Time: {1} s".format(i + 1, fusion_time))
        logger.info(f'融合图像时间：{fusion_time}\n\n')


        if i == 2:
            seg_start = time.time()
            train_seg(i, logger)
            print("|{0} Train Segmentation Model Sucessfully".format(i + 1))
            seg_end = time.time()
            seg_time = time.strftime("%H:%M:%S", time.gmtime(seg_end - seg_start))
            print("|{0} Train Segmentation Model Time: {1} s".format(i + 1, seg_time))
            logger.info(f'训练分割模型时间：{seg_time}\n\n')

    end_total = time.time()
    total_time = time.strftime(
        "%H:%M:%S", time.gmtime(end_total - start_total))
    print("Total Time: {0} s".format(total_time))
    logger.info(f'总时间：{total_time}\n\n')

    print("training Done!")