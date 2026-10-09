import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
import numbers
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import repeat
import math
from typing import Optional, Callable
from functools import partial
from einops.layers.torch import Rearrange
from thop import profile
import torch.utils.checkpoint as checkpoint


# input features: B x C x HW
# output: B x C x K (probabilities for each class)
# O(C * m * H * W)
class GumbelSoftmaxClustering(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(GumbelSoftmaxClustering, self).__init__()
        self.num_classes = num_classes
        self.centroids = nn.Parameter(torch.randn(num_classes, feature_dim))  # K x D

    def forward(self, features, tau=1.0, hard=False):  # features: B x C x HW
        # print("features shape:", features.shape)
        B, C, HW = features.shape

        # Normalize input for better distance computation
        features = F.normalize(features, dim=-1)  # (B, C, HW)
        centroids = F.normalize(self.centroids, dim=-1)  # (K, HW)

        # Compute squared Euclidean distance B x C x K
        dists = torch.cdist(features, centroids.unsqueeze(0).expand(B, -1, -1), p=2)  # B x C x K
        logits = -dists  # Inverse of distance

        # Gumbel-Softmax Sampling
        noise = torch.randn_like(logits)  # Sample noise from Gumbel distribution
        gumbel_noise = logits + noise
        probs = F.softmax(gumbel_noise / tau, dim=-1)  # Apply Gumbel-Softmax with temperature tau
        
        # If hard=True, return hard assignments (straight-through estimator)
        if hard:
            _, max_indices = torch.max(probs, dim=-1, keepdim=True)
            hard_probs = torch.zeros_like(probs).scatter_(-1, max_indices, 1.0)
            # straight-through estimator
            output = hard_probs - probs.detach() + probs
            return output
        else:
            return probs  # Soft probabilities for each class


# four directions Mamba
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        # print(self.expand) #! 0 ?
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    

    def forward(self, x: torch.Tensor, **kwargs):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B H W C
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()  # B C H W
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out.permute(0, 3, 1, 2)




# soft classifier along channel dimension : dont gate at input , but at output
# input: B x C x H x W
class SoftMamba_C(nn.Module):
    def __init__(self, num_classes,hw, in_channels, d_state, expand: float = 2., dropout: float = 0.):
        super(SoftMamba_C, self).__init__()
        self.num_classes = num_classes
        self.mamba_blocks = nn.ModuleList(
            [SS2D(d_model = in_channels, d_state = d_state,expand =  expand,dropout = dropout) for _ in range(num_classes)]
        )  # Initialize Mamba blocks
        self.soft_assignments = GumbelSoftmaxClustering(num_classes,hw)  # Soft assignments

    def forward(self, x):
        B, C, H, W = x.shape
        soft_assignments = self.soft_assignments(x.view(B,C,-1), tau=0.5, hard=False)  # B x C x num_classes
        outputs = []
        mamba_outputs = []
        for i in range(self.num_classes):
            mamba_output = self.mamba_blocks[i](x)  #! dont gate at input
            mamba_outputs.append(mamba_output)

        # 在输出处按soft assignment做加权
        weighted_outputs = []
        for i in range(self.num_classes):
            weight = soft_assignments[:, :, i].unsqueeze(-1).unsqueeze(-1)  # B x C x 1 x 1
            weighted_outputs.append(mamba_outputs[i] * weight)  #! gate here

        # Sum all weighted outputs
        return sum(weighted_outputs)  # B x C x H x W



class HardMamba_C(nn.Module):
    def __init__(self, num_classes, hw, in_channels, d_state, expand: float = 2., dropout: float = 0.):
        super(HardMamba_C, self).__init__()
        self.num_classes = num_classes
        self.mamba_blocks = nn.ModuleList([
            SS2D(d_model=in_channels, d_state=d_state, expand=expand, dropout=dropout)
            for _ in range(num_classes)
        ])
        self.router = GumbelSoftmaxClustering(num_classes, hw)

    def forward(self, x):  # x: B x C x H x W
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # B x C x HW

        # Step 1: Get hard assignments (B, C, K), then (B, C)
        soft_assignments = self.router(x_flat, tau=1.0, hard=True)  # one-hot
        masks = soft_assignments.permute(0, 2, 1)

        # Step 2: Apply masks to x
        x_expand = x.unsqueeze(1).expand(-1, self.num_classes, -1, -1, -1)  # B x K x C x H x W
        masks = masks.unsqueeze(-1).unsqueeze(-1)  # B x K x C x 1 x 1
        x_masked = x_expand * masks  # B x K x C x H x W

        # Step 4: Each expert processes its batch
        out_parts = []
        for k in range(self.num_classes):
            x_k = x_masked[:, k]  # B x C x H x W, masked channels only for class k
            out_k = self.mamba_blocks[k](x_k)
            out_parts.append(out_k)

        # Step 5: Add all expert outputs together
        out = sum(out_parts)  # B x C x H x W
        return out




class GumbelSoftmaxPixelClustering(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(GumbelSoftmaxPixelClustering, self).__init__()
        self.num_classes = num_classes
        self.centroids = nn.Parameter(torch.randn(num_classes, feature_dim))  # K x C

    def forward(self, features, tau=1.0, hard=False):  # features: B x C x H x W
        B, C, H, W = features.shape
        x = features.permute(0, 2, 3, 1).contiguous()  # B x H x W x C
        x_flat = x.view(B, -1, C)  # B x (H*W) x C

        # Normalize features and centroids
        x_norm = F.normalize(x_flat, dim=-1)  # B x (H*W) x C
        centroids_norm = F.normalize(self.centroids, dim=-1)  # K x C

        # Expand and compute pairwise distance
        dists = torch.cdist(x_norm, centroids_norm.unsqueeze(0).expand(B, -1, -1), p=2)  # B x (H*W) x K
        logits = -dists  # Inverse of distance

        # Gumbel-Softmax Sampling
        noise = torch.randn_like(logits)
        gumbel_noise = logits + noise
        probs = F.softmax(gumbel_noise / tau, dim=-1)  # B x (H*W) x K

        if hard:
            _, max_indices = torch.max(probs, dim=-1, keepdim=True)
            hard_probs = torch.zeros_like(probs).scatter_(-1, max_indices, 1.0)
            output = hard_probs - probs.detach() + probs  # straight-through
        else:
            output = probs

        return output.view(B, H, W, self.num_classes)  # B x H x W x K



if __name__ == "__main__":
    B, C, H, W = 2, 8, 16, 16
    num_classes = 4
    features = torch.randn(B, C, H, W)

    model = GumbelSoftmaxPixelClustering(num_classes, C)
    soft_out = model(features, tau=1.0, hard=False)
    hard_out = model(features, tau=1.0, hard=True)

    print("soft output shape:", soft_out.shape)  # [B, H, W, num_classes]
    print("hard output shape:", hard_out.shape)
    print("soft prob sum:", soft_out.sum(dim=-1).mean())
    print("hard prob unique values:", torch.unique(hard_out.sum(dim=-1)))