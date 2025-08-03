import os
import glob
import random
import numpy as np
from torch.utils import data

import utils.spk_utils as utils


class CSPK(data.Dataset):
    def __init__(self, root, dataset, n=41, patch_size=64, color_mode='rggb', fixed=True, ss=False):
        self.raws = glob.glob(os.path.join(root, 'spk', dataset, '*.dat'))
        self.gts = glob.glob(os.path.join(root, 'pl' if ss else 'gt', dataset, '*.npy'))
        self.raws.sort()
        self.gts.sort()
        self.patch_size = patch_size
        self.n = n
        self.color_mode = color_mode
        self.fixed = fixed

    def __len__(self): 
        return len(self.raws)

    def __getitem__(self, index):
        """
        return : ndarray
            cspk: (n, patch_size, patch_size)
            gt: (n, patch_size, patch_size, 3)
            mask: (4, patch_size, patch_size)
        """
        gt = np.load(self.gts[index])
        _, _, h, w  = gt.shape
        raw_f = open(self.raws[index], 'rb')
        raw = np.fromstring(raw_f.read(), 'B')
        raw_f.close()
        spk_seq = utils.raw2spike(raw, h, w)
        mask = self._generate_bayer_mask(h, w)
        cspk, gt, mask = self._preprocess_data(spk_seq, gt, mask)
        return cspk, gt, mask

    def _preprocess_data(self, spk_seq, gt, mask):
        if self.patch_size != 0:
            _, _, ih, iw  = gt.shape
            x = random.randrange(0, iw - self.patch_size + 1)
            y = random.randrange(0, ih - self.patch_size + 1)
            if self.fixed:
                x = x if x % 2 == 0 else x - 1
                y = y if y % 2 == 0 else y - 1
            spk_crop = spk_seq[:self.n, y:y+self.patch_size, x:x + self.patch_size] /1.
            gts_crop = gt[0, :, y:y+self.patch_size, x:x+self.patch_size] / 255.
            mask_crop = mask[:,y:y+self.patch_size, x:x+self.patch_size]
        else:
            spk_crop = spk_seq[:self.n, :, :] /1.
            gts_crop = gt[0] / 255.
            mask_crop = mask
        return spk_crop, gts_crop, mask_crop

    def _generate_bayer_mask(self, h, w):
        num = []
        flag = 0
        for c in self.color_mode:
            if c == 'r':
                num.append(0)
            elif c == 'g' and flag == 0:
                num.append(1)
                flag = 1
            elif c == 'g' and flag == 1:
                num.append(2)
            elif c == 'b':
                num.append(3)
        mask = np.zeros((4, h, w))
        rows_1 = slice(0, h, 2)
        rows_2 = slice(1, h, 2)
        cols_1 = slice(0, w, 2)
        cols_2 = slice(1, w, 2)
        mask[num[0], rows_1, cols_1] = 1
        mask[num[1], rows_1, cols_2] = 1
        mask[num[2], rows_2, cols_1] = 1
        mask[num[3], rows_2, cols_2] = 1
        return mask

class RCSPK(data.Dataset):
    def __init__(self, root, dataset, n=41, h=1000, w=1000, color_mode='rggb'): # bggr
        self.raws = glob.glob(os.path.join(root, 'spk', dataset, '*.dat'))
        self.raws.sort()
        self.n = n
        self.h = h
        self.w = w
        self.color_mode = color_mode

    def __len__(self): 
        return len(self.raws)

    def __getitem__(self, index):
        """
        return : ndarray
            cspk: (n, patch_size, patch_size)
            mask: (4, patch_size, patch_size)
        """
        raw_f = open(self.raws[index], 'rb')
        raw = np.fromstring(raw_f.read(), 'B')
        raw_f.close()
        spk_seq = utils.raw2spike(raw, self.h, self.w, upside_down=True)
        mask = self._generate_bayer_mask(self.h, self.w)
        return spk_seq, mask

    def _generate_bayer_mask(self, h, w):
        num = []
        flag = 0
        for c in self.color_mode:
            if c == 'r':
                num.append(0)
            elif c == 'g' and flag == 0:
                num.append(1)
                flag = 1
            elif c == 'g' and flag == 1:
                num.append(2)
            elif c == 'b':
                num.append(3)
        mask = np.zeros((4, h, w))
        rows_1 = slice(0, h, 2)
        rows_2 = slice(1, h, 2)
        cols_1 = slice(0, w, 2)
        cols_2 = slice(1, w, 2)
        mask[num[0], rows_1, cols_1] = 1
        mask[num[1], rows_1, cols_2] = 1
        mask[num[2], rows_2, cols_1] = 1
        mask[num[3], rows_2, cols_2] = 1
        return mask
