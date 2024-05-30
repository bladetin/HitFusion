# coding:utf-8
from multiprocessing.sharedctypes import Value
import os
import argparse
import time
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_TII import BiSeNet
from TaskFusion_dataset import Fusion_dataset
from new_fusenet import FusionNet
from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image

def main(modelpath):
    fusion_model_path = modelpath
    fusionmodel = FusionNet()
    fusionmodel.cuda()

    fusionmodel.load_state_dict(torch.load(fusion_model_path, map_location='cuda:4'))
    print('fusionmodel load done!')
    ir_path = '/Test_ir'
    vi_path = '/Test_vi_RGB'
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path)
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir,name) in enumerate(test_loader):

            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)

            image_shape = images_vis.shape
            image_h = image_shape[2]
            image_w = image_shape[3]
            pad_h = 16 - image_h % 16
            pad_w = 16 - image_w % 16
            images_vis = F.pad(images_vis, (0, pad_w, 0, pad_h), mode='reflect')
            images_ir = F.pad(images_ir, (0, pad_w, 0, pad_h), mode='reflect')

            images_vis = images_vis.cuda()
            images_ir = images_ir.cuda()
            images_vis_ycrcb = RGB2YCrCb(images_vis)
            logits = fusionmodel(images_vis_ycrcb, images_ir)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fusion_image = fusion_image[:, :, 0:image_h, 0:image_w]

            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)


def YCrCb2RGB(input_im):

    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)

    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()

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

def RGB2YCrCb(input_im):
    
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  
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
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Fusion with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='Fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=-1)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--fusionmodel', '-F', type=str, default='default') 
    args = parser.parse_args()

    if args.fusionmodel == 'default':
        args.fusionmodel = './Fusion/fusion_model.pth'
    else :
        args.fusionmodel = f'./Fusion/fusion_model_bak_{args.fusionmodel}.pth'

    n_class = 9
    seg_model_path = './Fusion/model_final.pth'
    fusion_model_path = './Fusion/fusionmodel_final.pth'
    fused_dir = './Fusion_results'
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

    start_time = time.time()
    main(args.fusionmodel)
    end_time = time.time()
    print('Fusion time: %f' % (end_time - start_time))
