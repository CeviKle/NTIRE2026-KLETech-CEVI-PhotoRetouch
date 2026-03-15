import math
import pathlib
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn

def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers.

    Args:
        shape (tuple): shape of image.
        ranges (tuple): range of coordinate value. Default: None.
        flatten (bool): flatten to (n, 2) or Not. Default: True.

    return:
        coord (Tensor): coordinates.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    coord = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        coord = coord.view(-1, coord.shape[-1])
    return coord

def get_wh_mgrid(width: int, height: int, dim: int = 2, flatten: bool = True) -> torch.Tensor:
    w = torch.linspace(-1, 1, steps=width)
    h = torch.linspace(-1, 1, steps=height)
    mgrid = torch.stack(torch.meshgrid(w, h), dim=-1)
    if flatten:
        return mgrid.reshape(-1, dim)
    else:
        return mgrid

def crop_image(img: np.ndarray, crop_size: int = 512):
    h, w, _ = img.shape
    ch, cw = (h - crop_size) // 2, (w - crop_size) // 2
    out_img = img[ch : ch + crop_size, cw : cw + crop_size]
    return out_img

class InputTensor(nn.Module):
    def __init__(self, input, gt, mask=None):
        super(InputTensor, self).__init__()
        self.input = input.float()
        self.gt = gt.float()
        self.mask = mask
        self.length = self.input.shape[0]

    def forward(self, xs):
        with torch.no_grad():
            xs = xs * torch.tensor([self.length], device=xs.device).float()
            indices = xs.long()
            indices.clamp(min=0, max=self.length - 1)
            if self.mask == None:
                return self.input[indices], self.gt[indices]
            else:
                return self.input[indices], self.gt[indices], self.mask[indices]
            

class InputWindTensor(nn.Module):
    def __init__(self, input, gt, flatten=True):
        super(InputWindTensor, self).__init__()
        self.input = input.float()
        self.gt = gt.float()
        self.length = self.input.shape[1]
        self.flatten = flatten

    def forward(self, xs):
        with torch.no_grad():
            xs = xs * torch.tensor([self.length], device=xs.device).float()
            indices = xs.long()
            indices.clamp(min=0, max=self.length - 1)
            if self.flatten:
                return self.input[:, indices].reshape((-1,self.input.shape[-1])), self.gt[:, indices].reshape((-1,self.gt.shape[-1]))
            else:
                return self.input[:, indices], self.gt[:, indices]
