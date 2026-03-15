import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss
from torch.autograd import Variable
from math import exp
import math

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = nn.Sequential(*list(vgg.children())[:16])  # up to relu3_3

        for param in self.layers.parameters():
            param.requires_grad = False

        self.layer_weights = layer_weights if layer_weights else 1.0

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, gt):
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
            gt = gt.repeat(1,3,1,1)

        x = (x - self.mean) / self.std
        gt = (gt - self.mean) / self.std

        feat_x = self.layers(x)
        feat_gt = self.layers(gt)

        loss = F.l1_loss(feat_x, feat_gt) * self.layer_weights
        return loss

import kornia.color as kc


class LabColorLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x, gt):
        x_lab = kc.rgb_to_lab(x)
        gt_lab = kc.rgb_to_lab(gt)

        loss = F.l1_loss(x_lab, gt_lab)
        return loss * self.weight
        

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
    
class L1LossColor(nn.Module):
    """L1 (mean absolute error, MAE) loss Regulized using color histogram of the input image.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', bins_num = 32):
        super(L1LossColor, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.region_size = 256 // bins_num

    def forward(self, pred, target, style_inp, hist, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        hist_regions = ((style_inp*255.0)/self.region_size).int()
        loss_weights = hist[hist_regions[:, 0], hist_regions[:, 1], hist_regions[:, 2]].unsqueeze(1)
        return self.loss_weight * l1_loss(
            pred, target, loss_weights, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(
            -(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) \
        for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=0, stride=window_size, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, stride=window_size, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=0, stride=window_size, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=0, stride=window_size, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=0, stride=window_size, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMWindLoss(torch.nn.Module):
    def __init__(self, window_size=12, size_average=True, loss_weight=1.0, INR_CNN=False):
        super(SSIMWindLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)
        self.loss_weight = loss_weight
        self.INR_CNN = INR_CNN

    def forward(self, samples1, samples2, windows_num):
        channel = samples1.size()[1]

        if channel == self.channel and \
                self.window.data.type() == samples1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if samples1.is_cuda:
                window = window.cuda(samples1.get_device())
            window = window.type_as(samples1)

            self.window = window
            self.channel = channel
        
        if not self.INR_CNN:
            # W2 x L x C
            samples1 = samples1.reshape((self.window_size**2, windows_num, channel))
            # C x W2 x L
            samples1 = samples1.permute((2, 0, 1))
            # C*W2 x L
            samples1 = samples1.reshape(-1, samples1.shape[-1])
            tmp_size = (self.window_size * int(math.sqrt(samples1.shape[-1])))
            # 1 x Cx W*L/2 x W*L^.5
            samples1 = torch.nn.functional.fold(samples1, tmp_size, self.window_size, dilation=1, padding=0, stride=self.window_size).unsqueeze(0)

            # W2 x L x C
            samples2 = samples2.reshape((self.window_size**2, windows_num, channel))
            # C x W2 x L
            samples2 = samples2.permute((2, 0, 1))
            # C*W2 x L
            samples2 = samples2.reshape(-1, samples2.shape[-1])
            tmp_size = (self.window_size * int(math.sqrt(samples2.shape[-1])))
            # 1 x Cx W*L/2 x W*L^.5
            samples2 = torch.nn.functional.fold(samples2, tmp_size, self.window_size, dilation=1, padding=0, stride=self.window_size).unsqueeze(0)

        # samples1 = torch.nn.functional.avg_pool2d(samples1, kernel_size=self.window_size, stride=self.window_size)
        # samples2 = torch.nn.functional.avg_pool2d(samples1, kernel_size=self.window_size, stride=self.window_size)

        # return torch.mean(torch.abs(samples1-samples2))


        return self.loss_weight * (1 - _ssim(samples1, samples2, window, self.window_size, channel, self.size_average))


class L_spa(nn.Module):

    def __init__(self, size_average=True, window_size=12, loss_weight=1.0, INR_CNN=False):
        super(L_spa, self).__init__()

        self.window_size = window_size
        self.loss_weight = loss_weight
        self.size_average = size_average
        self.INR_CNN = INR_CNN

        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, samples1 , samples2, windows_num):
        if not self.INR_CNN:
            # W2 x L x C
            (_, channel) = samples1.size()
            samples1 = samples1.reshape((self.window_size, self.window_size, windows_num, channel))
            # L x C x W2
            samples1 = samples1.permute((2, 3, 0, 1))

            samples2 = samples2.reshape((self.window_size, self.window_size, windows_num, channel))
            # L x C x W2
            samples2 = samples2.permute((2, 3, 0, 1))

        org_mean = torch.mean(samples1,1,keepdim=True)
        enhance_mean = torch.mean(samples2,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)

        if self.size_average:
            return torch.mean(E)
        else:
            E

class L_TV(nn.Module):
    def __init__(self, size_average=True, window_size=12, loss_weight=1.0, INR_CNN=False):
        super(L_TV,self).__init__()
        self.size_average = size_average
        self.TVLoss_weight = loss_weight
        self.window_size = window_size
        self.INR_CNN = INR_CNN

    def forward(self,samples, windows_num):
        if not self.INR_CNN:
            # W2 x L x C
            (_, channel) = samples.size()
            samples = samples.reshape((self.window_size, self.window_size, windows_num, channel))
            # L x C x W2
            samples = samples.permute((2, 3, 0, 1))

        batch_size = samples.shape[0]
        h_x = samples.shape[2]
        w_x = samples.shape[3]
        count_h =  (h_x-1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow((samples[:,:,1:,:]-samples[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((samples[:,:,:,1:]-samples[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    

class L_WeightedTV(nn.Module):
    def __init__(self, size_average=True, window_size=12, loss_weight=1.0, weighted=True, INR_CNN=False):
        super(L_WeightedTV,self).__init__()
        self.size_average = size_average
        self.TVLoss_weight = loss_weight
        self.window_size = window_size
        self.INR_CNN = INR_CNN

    def forward(self, inp_samples, samples, windows_num):
        if not self.INR_CNN:
            # W2 x L x C
            (_, channel) = samples.size()
            samples = samples.reshape((self.window_size, self.window_size, windows_num, channel))
            inp_samples = inp_samples[:, -3:].reshape((self.window_size, self.window_size, windows_num, channel))
            # L x C x W2
            samples = samples.permute((2, 3, 0, 1))
            inp_samples = inp_samples.permute((2, 3, 0, 1))

        batch_size = samples.shape[0]
        h_x = samples.shape[2]
        w_x = samples.shape[3]
        count_h =  (h_x-1) * w_x
        count_w = h_x * (w_x - 1)
        w_h = torch.exp(-1*torch.abs(inp_samples[:,:,1:,:]-inp_samples[:,:,:h_x-1,:]))
        w_w = torch.exp(-1*torch.abs(inp_samples[:,:,:,1:]-inp_samples[:,:,:,:w_x-1]))
        h_tv = torch.pow((samples[:,:,1:,:]-samples[:,:,:h_x-1,:]) * w_h, 2).sum()
        w_tv = torch.pow((samples[:,:,:,1:]-samples[:,:,:,:w_x-1]) * w_w, 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size