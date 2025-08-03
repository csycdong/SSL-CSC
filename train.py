import argparse
import numpy as np
import os

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
    train_dataset = CSPK(root=os.path.join(args.data_root, 'train'), dataset=args.train_set, n=args.n, patch_size=args.patch_size, ss=True) 
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)
    eval_dataset = CSPK(root=os.path.join(args.data_root, 'test'), dataset=args.eval_set, n=args.n, patch_size=0, ss=False)
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)
    model = Net(n=args.n)
    if args.gpu_num > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.gpu_num)])
    else:
        model = torch.nn.DataParallel(model)
    model.to(device)
    start_epoch = 0
    best_psnr = 0.0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=None if use_cuda  else 'cpu')
            start_epoch = checkpoint['epoch']
            best_psnr = checkpoint['best_psnr']
            model.load_state_dict(checkpoint['model_state'], strict=False)
            del checkpoint
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    # set optimizer, loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
    cspk_loss = torch.nn.L1Loss()
    print("Training has started.")
    for epoch in np.arange(start_epoch, args.epochs):
        # training
        train_loss = 0.0
        for train_index, (x, gt, mask) in enumerate(train_loader):
            model.train()
            x, gt, mask = x.to(device).float(), gt.to(device).float(), mask.to(device).float()
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(x, mask)
            loss = cspk_loss(output, gt)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (train_index + 1) % args.log_interval == 0 or train_index == len(train_loader) - 1 or train_index == 0:
                print("Epoch [%d/%d]" % (epoch + 1, args.epochs), "batch [%d/%d]:" % (train_index + 1, len(train_loader)),
                    "Loss =", train_loss / (train_index + 1), "    LR =", lr)
        model.eval()
        val_loss = 0.0
        psnr = 0.0
        ssim = 0.0
        for test_index, (x, gt, mask) in enumerate(eval_loader):
            x, gt, mask = x.to(device).float(), gt.to(device).float(), mask.to(device).float()
            with torch.no_grad():
                output = model(x, mask)
                loss = cspk_loss(output, gt)
                val_loss += loss.item()
                output = output.clamp(0., 1.).permute(0, 2, 3, 1).squeeze(0).cpu().numpy() *255.0
                gt = gt.permute(0, 2, 3, 1).squeeze(0).cpu().numpy() * 255.0
                tmp_psnr = compute_psnr(gt, output)
                tmp_ssim = compute_ssim(gt, output, data_range=255, multichannel=True)
                psnr += tmp_psnr
                ssim += tmp_ssim
                save_img_array(gt, os.path.join(result_path, '_'.join([str(epoch+1), str(test_index+1), 'gt.png'])), mode='RGB')
                save_img_array(output, os.path.join(result_path, '_'.join([str(epoch+1), str(test_index+1), 'output.png'])), mode='RGB')
                print("Epoch [%d/%d]" % (epoch + 1, args.epochs), "batch [%d/%d]:" % (test_index + 1, len(eval_loader)),
                "PSNR =", tmp_psnr, "    SSIM =", tmp_ssim)
        print("Train_Loss =", train_loss / len(train_loader), "    Test_Loss =", val_loss / len(eval_loader),
                "    PSNR =", psnr / len(eval_loader), "    SSIM =", ssim / len(eval_loader))
        # save checkpoint
        psnr = psnr / len(eval_loader)
        state = {'epoch': epoch + 1,
                'best_psnr': psnr,
                'model_state': model.state_dict()}
        torch.save(state, "{}/new_model.pth".format(save_path))
        if psnr >= best_psnr:
            best_psnr = psnr
            torch.save(state, "{}/best_model.pth".format(save_path))
        # update scheduler
        scheduler.step()
        print('Current best PSNR:', best_psnr)
    # end training
    print("Training has completed.")


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
                        help='root path of the dataset (default: data/SSR)')
    parser.add_argument('--log_path', nargs='?', type=str, default='logs',
                        help='path to save log information for the training (default: logs)')
    parser.add_argument('--result_path', nargs='?', type=str, default='results',
                        help='path to save test results (default: results)')
    parser.add_argument('--model_path', nargs='?', type=str, default='weights',
                        help='path to save trained model (default: weights)')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                         help='path to previous saved model to restart from')
    args = parser.parse_args()
    train(args)
