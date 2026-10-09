import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from basicsr.models.losses.loss_util import weighted_loss
from pytorch_msssim import ssim

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


#! old L1loss
# class L1Loss(nn.Module):
#     """L1 (mean absolute error, MAE) loss.

#     Args:
#         loss_weight (float): Loss weight for L1 loss. Default: 1.0.
#         reduction (str): Specifies the reduction to apply to the output.
#             Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
#     """

#     def __init__(self, loss_weight=1.0, reduction='mean'):
#         super(L1Loss, self).__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. '
#                              f'Supported ones are: {_reduction_modes}')

#         self.loss_weight = loss_weight
#         self.reduction = reduction

#     def forward(self, pred, target, weight=None, **kwargs):
#         """
#         Args:
#             pred (Tensor): of shape (N, C, H, W). Predicted tensor.
#             target (Tensor): of shape (N, C, H, W). Ground truth tensor.
#             weight (Tensor, optional): of shape (N, C, H, W). Element-wise
#                 weights. Default: None.
#         """
#         return self.loss_weight * l1_loss(
#             pred, target, weight, reduction=self.reduction)

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'.
        use_channel_weight (bool): 是否启用通道自适应加权. Default: False.
        alpha (float): 通道权重下限, 取值在 (0, 1). Default: 0.3.
        eps (float): 避免除零的极小数. Default: 1e-6.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 use_channel_weight=False,
                 alpha=0.3,
                 eps=1e-6):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

        # 新增参数
        self.use_channel_weight = use_channel_weight
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): (N, C, H, W)
            target (Tensor): (N, C, H, W)
            weight (Tensor, optional): (N, C, H, W)，外部像素权重（例如 mask）
        """
        # 不启用通道加权时，保持和原来完全一致
        if (not self.use_channel_weight) or pred.dim() != 4:
            return self.loss_weight * l1_loss(
                pred, target, weight, reduction=self.reduction)

        # ===== 启用通道加权的分支 =====
        # 先算基础的 L1 图 (N,C,H,W)，不做任何归约
        loss_map = F.l1_loss(pred, target, reduction='none')  # N,C,H,W

        # 如果外部有传入 weight（例如 mask），先乘上
        if weight is not None:
            loss_map = loss_map * weight

        # 只在 RGB 情况下做通道加权，其他通道数走普通 L1
        n, c, h, w = loss_map.shape
        if c != 3:
            if self.reduction == 'mean':
                loss = loss_map.mean()
            elif self.reduction == 'sum':
                loss = loss_map.sum()
            else:
                loss = loss_map
            return self.loss_weight * loss

        # 计算每个样本、每个通道的平均误差 e_c (基于当前 batch)
        # 形状: (N, 3)
        with torch.no_grad():
            e = loss_map.mean(dim=(2, 3))  # N,C
            mean_e = e.mean(dim=1, keepdim=True)  # N,1
            # 相对误差，三个通道的平均值大约为 1
            rel = e / (mean_e + self.eps)        # N,C
            # 通道权重: w_c = alpha + (1-alpha)*rel_c
            # 这样平均权重大约为 1，误差大的通道权重略大，误差小的略小
            chan_w = self.alpha + (1.0 - self.alpha) * rel      # N,C
            # 不让梯度回到权重里
            chan_w = chan_w.unsqueeze(-1).unsqueeze(-1)  # N,C,1,1

        # 应用通道权重到 loss_map
        # chan_w 在 no_grad 里算的，本身不在图里，相当于常数权重
        loss_map = loss_map * chan_w

        # 按 reduction 归约
        if self.reduction == 'mean':
            loss = loss_map.mean()
        elif self.reduction == 'sum':
            loss = loss_map.sum()
        else:
            loss = loss_map

        return self.loss_weight * loss




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

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


# class SSIMLoss(nn.Module):
#     def __init__(self,loss_weight=1, window_size=11, size_average=True):
#         super(SSIMLoss, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.loss_weight = loss_weight

#     def forward(self, x, y):
#         ssim_loss = 1 - ssim(x, y, data_range=1.0, size_average=self.size_average)
#         return self.loss_weight * ssim_loss


class SSIMLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 window_size=11,
                 size_average=True,
                 use_channel_weight=False,  # 新增: 是否启用通道加权
                 alpha=0.3,                 # 新增: 通道权重下限
                 eps=1e-6):                 # 新增: 防止除零
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.loss_weight = loss_weight

        self.use_channel_weight = use_channel_weight
        self.alpha = alpha
        self.eps = eps

    def forward(self, x, y):
        """
        x, y: (N, C, H, W), 范围 [0,1]
        """
        # 不启用通道加权，完全保持原始行为
        if (not self.use_channel_weight) or x.dim() != 4 or x.size(1) != 3:
            ssim_loss = 1 - ssim(
                x, y, data_range=1.0, size_average=self.size_average
            )
            return self.loss_weight * ssim_loss

        # ===== 启用通道加权的分支 =====
        channel_losses = []
        # 逐通道计算 SSIM loss
        for c in range(3):
            xc = x[:, c:c+1, ...]
            yc = y[:, c:c+1, ...]
            ssim_val = ssim(
                xc, yc, data_range=1.0, size_average=self.size_average
            )
            # 无论返回的是标量还是向量, 都压成标量
            loss_c = (1 - ssim_val).mean()
            channel_losses.append(loss_c)

        losses = torch.stack(channel_losses)   # 形状: (3,)

        # 用 detach 的误差来算权重, 避免梯度流入权重本身
        with torch.no_grad():
            e = losses.detach()               # (3,)
            mean_e = e.mean()
            rel = e / (mean_e + self.eps)     # 相对误差, 三通道平均约为 1
            # 通道权重: w_c = alpha + (1-alpha) * rel_c
            # 误差大的通道权重大一些, 但每个通道至少有 alpha
            w = self.alpha + (1.0 - self.alpha) * rel  # (3,)

        # 加权求和得到最终 SSIM loss
        total_loss = (w * losses).sum()

        return self.loss_weight * total_loss


class EdgeLoss(nn.Module):
    def __init__(self,loss_weight = 1.0):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        self.target_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.loss_weight = loss_weight

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff
    
    def forward(self, x, y):
        _, _, x_kw, x_kh = x.shape
        _, _, y_kw, y_kh = y.shape
        if x_kw == y_kw:
            loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        else:
            loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(self.target_down(y)))
        # loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.loss_weight


import torch
import torch.nn as nn
import torch.nn.functional as F


def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    # rgb: [0, 1]
    # piecewise, differentiable almost everywhere
    threshold = 0.04045
    low = rgb / 12.92
    high = ((rgb + 0.055) / 1.055).pow(2.4)
    return torch.where(rgb <= threshold, low, high)


def _rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    # rgb: (N, 3, H, W) in [0, 1], sRGB, D65
    rgb_lin = _srgb_to_linear(rgb)

    r = rgb_lin[:, 0:1, :, :]
    g = rgb_lin[:, 1:2, :, :]
    b = rgb_lin[:, 2:3, :, :]

    # sRGB to XYZ (D65)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return torch.cat([x, y, z], dim=1)


def _xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    # xyz: (N, 3, H, W), D65
    # Reference white for D65
    xn = 0.95047
    yn = 1.0
    zn = 1.08883

    x = xyz[:, 0:1, :, :] / xn
    y = xyz[:, 1:2, :, :] / yn
    z = xyz[:, 2:3, :, :] / zn

    delta = 6.0 / 29.0
    delta3 = delta ** 3
    inv_3delta2 = 1.0 / (3.0 * (delta ** 2))

    def f(t: torch.Tensor) -> torch.Tensor:
        # t > delta^3: t^(1/3)
        # else: t/(3*delta^2) + 4/29
        t_cbrt = torch.clamp(t, min=1e-12).pow(1.0 / 3.0)
        t_lin = t * inv_3delta2 + (4.0 / 29.0)
        return torch.where(t > delta3, t_cbrt, t_lin)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([L, a, b], dim=1)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    # rgb: (N,3,H,W), expected in [0,1]
    xyz = _rgb_to_xyz(rgb)
    lab = _xyz_to_lab(xyz)
    return lab


class LabABLoss(nn.Module):
    """Lab a,b loss for color supervision.

    Args:
        loss_weight (float): loss weight. Default: 1.0.
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'.
        loss_type (str): 'l1' or 'mse'. Default: 'l1'.
        clamp_input (bool): clamp pred/target to [0,1] before conversion. Default: True.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', loss_type='l1', clamp_input=True):
        super(LabABLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')
        if loss_type not in ['l1', 'mse']:
            raise ValueError(f'Unsupported loss_type: {loss_type}. Use "l1" or "mse".')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_type = loss_type
        self.clamp_input = clamp_input

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred:   (N,3,H,W) RGB, expected in [0,1] (or will be clamped if clamp_input=True)
            target: (N,3,H,W) RGB, expected in [0,1]
            weight: optional element-wise weight, shape (N,1,H,W) or (N,2,H,W) or (N,3,H,W)
        """
        if self.clamp_input:
            pred = pred.clamp(0.0, 1.0)
            target = target.clamp(0.0, 1.0)

        lab_pred = rgb_to_lab(pred)
        lab_tgt = rgb_to_lab(target)

        ab_pred = lab_pred[:, 1:3, :, :]
        ab_tgt = lab_tgt[:, 1:3, :, :]

        if self.loss_type == 'l1':
            diff = (ab_pred - ab_tgt).abs()
        else:
            diff = (ab_pred - ab_tgt) ** 2

        if weight is not None:
            # make weight broadcastable to (N,2,H,W)
            if weight.shape[1] == 1:
                w = weight.repeat(1, 2, 1, 1)
            elif weight.shape[1] >= 2:
                w = weight[:, 0:2, :, :]
            else:
                raise ValueError('weight has invalid channel dim.')
            diff = diff * w

        if self.reduction == 'none':
            loss = diff
        elif self.reduction == 'sum':
            loss = diff.sum()
        else:
            loss = diff.mean()

        return self.loss_weight * loss
