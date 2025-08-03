import argparse
import numpy as np
import os

import cv2
from PIL import Image
import lpips
import torch
from torch.utils import data
import torch.nn.functional as F

from models.spike2flow.spike2flow import Spike2Flow
from utils.spk_utils import interval_torch
from utils.warp_utils import flow_warp
from utils.cspk_dataset import CSPK
from utils.metrics import compute_psnr, compute_ssim
from utils.data_utils import save_img_array


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # check GPU number
    gpu_num = torch.cuda.device_count()
    if args.gpu_num > gpu_num:
        args.gpu_num = gpu_num
        print('GPU number has been ajusted to', gpu_num)
    use_cuda = (args.gpu_num != 0) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print('Use GPU:')
        for i in range(args.gpu_num):
            print(torch.cuda.get_device_name(i))
    # update batch size according to GPU number
    result_path = os.path.join(args.data_root, args.tag, 'pl', args.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # set dataloader
    dataset = CSPK(root=os.path.join(args.data_root, args.tag), dataset=args.dataset, n=args.n, patch_size=0, color_mode=args.color_mode)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    # load the model
    model = Spike2Flow()
    if args.gpu_num > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.gpu_num)])
    else:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.resume))
    model.to(device)
    model.eval()
    loss_fn = lpips.LPIPS(net='alex').cuda()
    psnr = 0.0
    ssim = 0.0
    lpips_ = 0.0
    for test_index, (x, gt, mask) in enumerate(dataloader):
        padding_size = 8
        row_pad = padding_size - x.shape[2] % padding_size if x.shape[2] % padding_size != 0 else 0
        col_pad = padding_size - x.shape[3] % padding_size if x.shape[3] % padding_size != 0 else 0
        x =  torch.Tensor(np.pad(x, ((0, 0), (0, 0), (0, row_pad), (0, col_pad)), 'edge')).to(device).float() # b n h w
        mask = torch.Tensor(np.pad(mask, ((0, 0), (0, 0), (0, row_pad), (0, col_pad)), 'constant', constant_values=0)).to(device).float() # b 4 h w
        gt = gt.to(device).float() # b n h w 3
        x, gt, mask = x.to(device).float(), gt.to(device).float(), mask.to(device).float()
        with torch.no_grad():
            inters = interval_torch(x)
            r = F.max_pool2d(inters*mask[:,0:1,:,:], kernel_size=(2, 2), stride=(2, 2), padding=0)
            g = F.avg_pool2d(inters*(mask[:,1:2,:,:]+mask[:,2:3,:,:]), kernel_size=(2, 2), stride=(2, 2), padding=0) / 2
            b = F.max_pool2d(inters*mask[:,3:4,:,:], kernel_size=(2, 2), stride=(2, 2), padding=0)
            flow_l_r = model(torch.flip(r[:,1:31,:,:], dims=[1]))[2][-1]
            flow_l_g = model(torch.flip(g[:,1:31,:,:], dims=[1]))[2][-1]
            flow_l_b = model(torch.flip(b[:,1:31,:,:], dims=[1]))[2][-1]
            flow_l = (flow_l_r + flow_l_g + flow_l_b) / 3
            flow_l = F.interpolate(flow_l, scale_factor=2, mode='bilinear', align_corners=False) * 2 / 9
            flow_r_r = model(r[:,10:40,:,:])[2][-1]
            flow_r_g = model(g[:,10:40,:,:])[2][-1]
            flow_r_b = model(b[:,10:40,:,:])[2][-1]
            flow_r = (flow_r_r + flow_r_g + flow_r_b) / 4
            flow_r = F.interpolate(flow_r, scale_factor=2, mode='bilinear', align_corners=False) * 2 / 9
            itensity = 1. / inters
            r = upsampling(itensity, mask[:,0:1,:,:], flow_l, flow_r, args.clip)
            g = upsampling(itensity, mask[:,1:2,:,:] + mask[:,2:3,:,:], flow_l, flow_r, args.clip, coef=2)
            b = upsampling(itensity, mask[:,3:4,:,:], flow_l, flow_r, args.clip)
            output = torch.cat([r, g, b], axis=1)[:,:,:gt.shape[2],:gt.shape[3]] / args.coeff
            tmp_lpips = loss_fn(gt, output).item()
            output = output.clamp(0., 1.).cpu().numpy() * 255.0
            output = output.transpose((0, 2, 3, 1))[0]
            gt = gt.permute(0, 2, 3, 1).squeeze(0).cpu().numpy() * 255.0
            tmp_psnr = compute_psnr(gt, output)
            tmp_ssim = compute_ssim(gt, output, data_range=255, multichannel=True)
            psnr += tmp_psnr
            ssim += tmp_ssim
            lpips_ += tmp_lpips
            save_img_array(output, os.path.join('results', args.dataset, "{:04d}.png".format(test_index)), mode='RGB')
            print("batch [%d/%d]:" % (test_index + 1, len(dataloader)), "PSNR =", tmp_psnr, "    SSIM =", tmp_ssim,  "    LPIPS = ", tmp_lpips)
    print("PSNR =", round(psnr / len(dataloader), 2), "    SSIM =", round(ssim / len(dataloader), 4),  "    LPIPS =", round(lpips_ / len(dataloader), 4))


def upsampling(x, mask, of_l, of_r, clip=7, coef=4):
    x = x * mask
    n = x.shape[1]
    mid = n // 2
    x_supp = F.avg_pool2d(x, kernel_size=2, stride=2)*coef
    x_supp = F.interpolate(x_supp, scale_factor=2, mode='bicubic', align_corners=False)
    aligned_supp = None
    aligned_x = None
    aligned_mask = None
    for i in range(n-clip):
        if i < clip:
            continue
        cur_supp = x_supp[:, i:i+1, :,:]
        cur_x = x[:, i:i+1, :,:]
        cur_mask = mask
        if i != mid:
            of = (mid - i) * of_l if i < mid else (i - mid) * of_r
            cur_supp = flow_warp(cur_supp, of)
            cur_x = flow_warp(cur_x, of)
            cur_mask = flow_warp(mask, of)
        aligned_supp = cur_supp if i == clip else torch.cat([aligned_supp, cur_supp], axis=1)
        aligned_x = cur_x if i == clip else torch.cat([aligned_x, cur_x], axis=1)
        aligned_mask = cur_mask if i == clip else torch.cat([aligned_mask, cur_mask], axis=1)
    supp = torch.mean(aligned_supp, axis=1, keepdim=True)
    x_mask = torch.sum(aligned_mask, axis=1, keepdim=True)
    x_up = torch.sum(aligned_x, axis=1, keepdim=True) / x_mask
    nan_mask = torch.isnan(x_up).type(torch.int)
    x_up[torch.isnan(x_up)] = 0
    x_up = x_up + supp * nan_mask
    return x_up


def upsampling2(x, mask, of_l, of_r, clip=7):
    x = x * mask
    n = x.shape[1]
    mid = n // 2
    aligned_x = None
    aligned_mask = None
    for i in range(n-clip):
        if i < clip:
            continue
        cur_x = x[:, i:i+1, :,:]
        cur_mask = mask
        if i != mid:
            of = (mid - i) * of_l if i < mid else (i - mid) * of_r
            cur_x = flow_warp(cur_x, of)
            cur_mask = flow_warp(mask, of)
        aligned_x = cur_x if i == clip else torch.cat([aligned_x, cur_x], axis=1)
        aligned_mask = cur_mask if i == clip else torch.cat([aligned_mask, cur_mask], axis=1)
    x_up = torch.sum(aligned_x, axis=1, keepdim=True) / torch.sum(aligned_mask, axis=1, keepdim=True)
    x_up[torch.isnan(x_up)] = 0
    return x_up


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training")    
    parser.add_argument('--gpu_id', type=str, default='0',  help='the id of GPU.') 
    parser.add_argument('--n', type=int, default=41, 
                        help='frame number of training spike streams (default: 41)')
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='gpu number for training, 0 means cpu training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='num_workers parameter for dataloader (default: 4)')
    parser.add_argument('--pin_memory', nargs='?', type=bool, default=True,
                        help='pin_memory parameter for data_loader (default: True)')
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--color_mode', type=str, default='rggb')
    parser.add_argument('--dataset', type=str, default='REDS120', help='dataset (default: REDS120)')
    parser.add_argument('--data_root', nargs='?', type=str, default='data/SSR',
                        help='root path of the dataset (default: data/SSR)')
    parser.add_argument('--resume', nargs='?', type=str, default='weights/spike2flow.pth',
                         help='path to previous saved model to restart from')
    parser.add_argument('--coeff', type=float, default=0.6)
    parser.add_argument('--clip', type=int, default=7) 
    args = parser.parse_args()
    main(args)
