import argparse
import numpy as np
import os

import lpips
import time
import torch
from torch.utils import data

from utils.cspk_dataset import CSPK
from utils.metrics import compute_psnr, compute_ssim
from utils.data_utils import save_img_array
from models.network import Net as Net


def train(args):
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
    batch_size = args.batch_size
    if use_cuda:
        batch_size = batch_size * args.gpu_num
    # set paths
    save_path = os.path.join(args.model_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_path = os.path.join(args.result_path, args.eval_set)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # set dataloader
    eval_dataset = CSPK(root=os.path.join(args.data_root, 'test'), dataset=args.eval_set, n=args.n, patch_size=0, ss=False, color_mode=args.color_mode)
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)
    # load the model
    model = Net(n=args.n)
    if args.gpu_num > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.gpu_num)])
    else:
        model = torch.nn.DataParallel(model)
    model.to(device)
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=None if use_cuda  else 'cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state'], strict=False)
            del checkpoint
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    loss_fn = lpips.LPIPS(net='alex').cuda()
    model.eval()
    psnr = 0.0
    ssim = 0.0
    lpips_ = 0.0
    time_ = 0.0
    for test_index, (x, gt, mask) in enumerate(eval_loader):
        x, gt, mask = x.to(device).float(), gt.to(device).float(), mask.to(device).float()
        with torch.no_grad():
            start_time = time.time()
            output = model(x, mask)
            tmp_lpips = loss_fn(gt, output).item()
            tmp_time = time.time() - start_time
            output = output.clamp(0., 1.).permute(0, 2, 3, 1).squeeze(0).cpu().numpy() *255.0
            gt = gt.permute(0, 2, 3, 1).squeeze(0).cpu().numpy() * 255.0
            tmp_psnr = compute_psnr(gt, output)
            tmp_ssim = compute_ssim(gt, output, data_range=255, multichannel=True)
            psnr += tmp_psnr
            ssim += tmp_ssim
            lpips_ += tmp_lpips
            time_ += tmp_time
            save_img_array(gt, os.path.join(result_path, '_'.join([str(test_index+1), 'gt.png'])), mode='RGB')
            save_img_array(output, os.path.join(result_path, '_'.join([str(test_index+1), 'output.png'])), mode='RGB')
            print("batch [%d/%d]:" % (test_index + 1, len(eval_loader)), "PSNR =", tmp_psnr, "    SSIM =", tmp_ssim,  "    LPIPS = ", tmp_lpips,  "    Time = ", tmp_time)
    print("PSNR =", round(psnr / len(eval_loader), 2), "    SSIM =", round(ssim / len(eval_loader), 4), "    LPIPS =", round(lpips_ / len(eval_loader), 4), "    Time =", round(time_ / len(eval_loader), 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training")    
    parser.add_argument('--loss', type=str, default='') 
    parser.add_argument('--gpu_id', type=str, default='0',  help='the id of GPU.') 
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='bath size for the training (default: 4)')
    parser.add_argument('--patch_size', type=int, default=128, 
                        help='patch size for the training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate for training (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.9, 
                        help='step lr param gamma (default: 0.9. If 1, means no decay)')
    parser.add_argument('--lr_step', type=int, default=25,
                        help='learning rate decay interval (default: 1)')
    parser.add_argument('--epochs', type=int, default=5000, 
                        help='number of epochs to train (default: 5000)')
    parser.add_argument('--n', type=int, default=41, 
                        help='frame number of training spike streams (default: 41)')
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='gpu number for training, 0 means cpu training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='num_workers parameter for dataloader (default: 4)')
    parser.add_argument('--pin_memory', nargs='?', type=bool, default=True,
                        help='pin_memory parameter for data_loader (default: True)')
    parser.add_argument('--no_tensor_board', action='store_true', default=False,
                        help='dont\'t use tensor_board (default: False)')
    parser.add_argument('--log_interval', type=int, default=5, 
                        help='iteration interval for printing logs (default: 5)')
    parser.add_argument('--eval_interval', type=int, default=200, 
                        help='iteration interval for evaluation (default: 200)')
    parser.add_argument('--eval_set', type=str, default='test',
                        help='dataset for evaluation (default: test)')
    parser.add_argument('--train_set', type=str, default='REDS120',
                        help='dataset for evaluation (default: REDS120)')
    parser.add_argument('--data_root', nargs='?', type=str, default='data/SSR',
                        help='root path of the dataset (default: .data/SSR)')
    parser.add_argument('--log_path', nargs='?', type=str, default='logs',
                        help='path to save log information for the training (default: logs)')
    parser.add_argument('--result_path', nargs='?', type=str, default='results',
                        help='path to save test results (default: results)')
    parser.add_argument('--model_path', nargs='?', type=str, default='weights',
                        help='path to save trained model (default: weights)')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                         help='path to previous saved model to restart from')
    parser.add_argument('--color_mode', type=str, default='rggb')
    args = parser.parse_args()
    train(args)
