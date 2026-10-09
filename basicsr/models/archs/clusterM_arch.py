# The Code Implementatio of MambaIR model for Real Image Denoising task
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange
import math
from typing import Optional, Callable
from einops import rearrange, repeat
from functools import partial
from thop import profile
import time
from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
# from .segFormer import get_segment_prob
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
NEG_INF = -1000000
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from collections import defaultdict
###########################################################################
class RM_decoder0(nn.Module):
    def __init__(self, num_classes,hw, in_channels, d_state , expand: float = 2., dropout: float = 0.,
            isTest=False):
        super(RM_decoder0, self).__init__()
        self.num_classes = num_classes
        self.tau_threshold = 0.5
        self.mamba_blocks = nn.ModuleList(
            [SS2D_dh0(d_model = in_channels, d_state = d_state,expand = expand,dropout = dropout) 
                for _ in range(num_classes)]
        )  # Initialize Mamba blocks
        self.isTest = isTest
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 可学习标量参

    def forward(self, x, soft_assignment, h_global):
        """
        force_expert_id (int): 0, 1, 2...
        """
        self.prob = soft_assignment
        B, H, W, C = x.shape
        
        x_spatial = x
        out = torch.zeros_like(x_spatial)

        x_flat = x_spatial.view(B, H * W, C)
        hard_flat = soft_assignment.view(B, H * W, self.num_classes)

        for i in range(self.num_classes):
            mask_i = hard_flat[..., i]
            if not mask_i.any():
                continue

            for b in range(B):
                mb = mask_i[b]                     # HW
                if not mb.any():
                    continue

                idx = mb.nonzero(as_tuple=False).squeeze(-1)   # L
                selected_feat = x_flat[b, idx]     # L, C
                L = selected_feat.shape[0]

                side = int(math.ceil(L ** 0.5))
                pad_len = side * side - L
                if pad_len > 0:
                    selected_feat = F.pad(
                        selected_feat, (0, 0, 0, pad_len), value=0.0
                    )                               # (side*side, C)

                reshaped = selected_feat.view(1, side, side, C)

                mamba_out = self.mamba_blocks[i](reshaped,h_global[b]).view(-1, C)[:L]

                h_idx = idx // W
                w_idx = idx % W
                out[b, h_idx, w_idx] += mamba_out

        out = out + self.alpha * x
        return out

# b h w c
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def l2n(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def orthonormal_rows(k, r, device):
    a = torch.randn(r, k, device=device)
    q, _ = torch.linalg.qr(a, mode="reduced")  # r×k，
    p = q.T                                    # k×r
    return p

@torch.no_grad()
def _init_kpp(x, k):
    # x: B×N×R
    B, N, R = x.shape
    idx = torch.randint(0, N, (B, 1), device=x.device)
    centers = x[torch.arange(B)[:,None], idx]  # B×1×R
    for _ in range(1, k):
        dist2 = (x[:, :, None, :] - centers[:, None, :, :]).pow(2).sum(-1).min(dim=2).values  # B×N
        prob = dist2 / (dist2.sum(dim=1, keepdim=True) + 1e-6)
        new_idx = torch.multinomial(prob, 1)  # B×1
        new_c = x[torch.arange(B)[:,None], new_idx]  # B×1×R
        centers = torch.cat([centers, new_c], dim=1)
    return centers  # B×K×R


def get_kcenters_perBatch(pooled, num_clusters, iters=5, temp=0.25, use_soft=False):
    B, H, W, R = pooled.shape
    X = l2n(pooled.reshape(B, H*W, R), dim=-1)         # B×N×R
    N = H * W
    with torch.no_grad():
        C = _init_kpp(X, num_clusters)                # B×K×R

    for _ in range(iters):
        dist2 = (X[:, :, None, :] - C[:, None, :, :]).pow(2).sum(-1)  # B×N×K
        if use_soft:
            w = F.softmax(-dist2 / temp, dim=-1).detach()             # stop-grad 
        else:
            idx = dist2.argmin(dim=-1)                                # B×N
            w = F.one_hot(idx, num_classes=num_clusters).float().detach()  # B×N×K
        num = torch.einsum("bnk,bnr->bkr", w, X)                      # B×K×R
        den = w.sum(dim=1, keepdim=False).clamp_min(1.0)              # B×K
        C = l2n(num / den.unsqueeze(-1), dim=-1)                      # B×K×R

    hard_idx = None if use_soft else idx
    return C, hard_idx

def greedy_max_match(sim):  # sim: K×K
    K = sim.shape[0]
    used_c = torch.zeros(K, dtype=torch.bool, device=sim.device)
    used_p = torch.zeros(K, dtype=torch.bool, device=sim.device)
    order = []
    flat = sim.reshape(-1)
    vals, inds = torch.sort(flat, descending=True)
    for idx in inds.tolist():
        c = idx // K
        p = idx % K
        if not used_c[c] and not used_p[p]:
            used_c[c] = True
            used_p[p] = True
            order.append((p, c))
        if len(order) == K:
            break
    order = sorted(order, key=lambda t: t[0])  
    chosen_centers = torch.tensor([c for _, c in order], device=sim.device, dtype=torch.long)
    return chosen_centers  

def get_expert_assignments_new(k_centers, pk, pooled, tau=1.0):
    """
    k_centers: B × K × R 
    pk:        K × R     
    pooled:    B × H × W × R （
    """
    B, H, W, R = pooled.shape
    K = pk.shape[0]
    pk_n = l2n(pk, dim=-1)                  # K×R
    match_idx_list = []
    k_sorted_list = []

    # --- Step 1: Sort (Align C to P) ---
    for b in range(B):
        cb = l2n(k_centers[b], dim=-1)      # K×R
        sim = torch.matmul(cb, pk_n.t())    # K×K (similarity)
        chosen = greedy_max_match(sim)      
        k_sorted_list.append(cb[chosen])    
        match_idx_list.append(chosen)

    k_centers_sorted = torch.stack(k_sorted_list, dim=0)          # B×K×R
    match_idx = torch.stack(match_idx_list, dim=0)                # B×K

    # --- Step 2: Route (Assign X to C_sorted) ---
    X = l2n(pooled, dim=-1)                                       # B×H×W×R
    logits = torch.einsum("bhwr,bkr->bhwk", X, k_centers_sorted)  
    
    expert_assign = F.softmax(logits / tau, dim=-1)               # B×H×W×K
    index = expert_assign.argmax(dim=-1)
    hard_one_hot = F.one_hot(index, K).float()
    expert_assign_ste = (hard_one_hot - expert_assign).detach() + expert_assign

    return k_centers_sorted, expert_assign_ste, expert_assign


#! return hidden state - ss2d
# input  : b h w c 
# but the input of forward_core is b c h w
# output : b h w c 
class SS2D_eh0(nn.Module):
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
        self.global_proj = nn.Linear(self.d_inner * 4, d_model, bias=False)

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

        out_y, last_state = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=True,
        )

        last_state = last_state.view(B, K, self.d_inner, -1)   # (B, K, d_inner, d_state)
        h_global = last_state.mean(dim=1)
    
        out_y = out_y.view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y , h_global

    def forward(self, x: torch.Tensor, **kwargs):
        # print(x.shape)
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 , h_global = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        # print("out shape:", out.shape)
        return out , h_global


#! use hidden state - ss2d
# input  : b h w c 
# but the input of forward_core is b c h w
# output : b h w c 
class SS2D_dh0(nn.Module):
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


        self.FiLM = nn.Sequential(
            nn.Linear(d_state, self.d_inner),
            nn.SiLU(),
            nn.Linear(self.d_inner, 2)
        )

        last = self.FiLM[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

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

    def fusion_lastState(self, y, h_global):
        # h_global shape : C State
        _,_,_,c = y.shape
        # Treat encoder context as a fixed cue, independently of scan backend.
        gamma_beta = self.FiLM(h_global.detach())          # (C, 2)
        gamma = gamma_beta[:, 0]              # (C,)
        beta  = gamma_beta[:, 1]              # (C,)
        gamma = gamma.view(1, 1, 1, c)        # (1,1,1,C)
        beta  = beta.view(1, 1, 1, c)         # (1,1,1,C)

        # FiLM 
        y = y * (1.0 + gamma) + beta          # (B,H,W,C)
        return y
    
    def forward(self, x: torch.Tensor, h_global, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.fusion_lastState(y, h_global)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out 



class DynamicsTransferRouter(nn.Module):
    def __init__(self, in_channels, dt_rank_source, router_dim, lora_r=4):
        super().__init__()
        self.dt_rank_source = dt_rank_source
        self.num_directions = 4  # SS2D 固定的 4 个方向
        self.router_dim = router_dim
        
        # === 1. Prior  ===
        self.total_sensors = self.num_directions * self.dt_rank_source
        self.prior_fusion = nn.Linear(self.total_sensors, self.router_dim, bias=False)
        
        # === 2. LoRA  ===
        self.lora_r = lora_r
        self.lora_A = nn.Linear(in_channels, lora_r, bias=False)
        self.lora_B = nn.Linear(lora_r, self.router_dim, bias=False)
        
        self.lora_alpha = nn.Parameter(torch.tensor(1.0))

        # === 3. init ===
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.normal_(self.lora_B.weight, std=0.01)

        nn.init.kaiming_uniform_(self.prior_fusion.weight, a=math.sqrt(5))

    def forward(self, x, encoder_weight_raw):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        L = H * W
        
        x_img = x.permute(0, 3, 1, 2).contiguous() # (B, C, H, W)
        
        x_n = x_img.view(B, C, -1) # (B, C, L)
        x_t = torch.transpose(x_img, 2, 3).contiguous().view(B, C, -1)
        
        x_hwwh = torch.stack([x_n, x_t], dim=1)
        
       
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        
       
        xs = xs.permute(0, 1, 3, 2).contiguous()

      
        w_enc = encoder_weight_raw.detach() # (K, Out, Enc_C)
        w_dt = w_enc[:, :self.dt_rank_source, :] # (K, dt, Enc_C)
        
       
        if w_dt.shape[-1] >= C:
             w_dt = w_dt[:, :, :C]
        else:
             w_dt = F.pad(w_dt, (0, C - w_dt.shape[-1]))
             
        w_dt = w_dt.permute(0, 2, 1)

       
        prior_raw = torch.matmul(xs, w_dt)

        y0, y1, y2, y3 = prior_raw.chunk(4, dim=1)
        y0 = y0.squeeze(1) # (B, L, dt)
        y1 = y1.squeeze(1)
        y2 = y2.squeeze(1)
        y3 = y3.squeeze(1)

       
        out0 = y0.view(B, H, W, -1)  
        out1 = y1.view(B, W, H, -1).transpose(1, 2)
        out2 = torch.flip(y2, dims=[1]).view(B, H, W, -1)
        out3 = torch.flip(y3, dims=[1]).view(B, W, H, -1).transpose(1, 2)
        prior_aligned = torch.cat([out0, out1, out2, out3], dim=-1)

       
        prior_feat = self.prior_fusion(prior_aligned)
        lora_feat = self.lora_A(x)
        lora_feat = self.lora_B(lora_feat)

        total_feat = F.softplus(prior_feat + self.lora_alpha * lora_feat)
        
        return total_feat


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
# from [b c h w] to [b c h w]
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x) # b c h w
        x = rearrange(x, "b c h w -> b h w c").contiguous()
        return x

class Downsample2(nn.Module):
    def __init__(self, n_feat):
        super(Downsample2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # b h w c  -> b c h w
        x = self.body(x)
        x = x.permute(0,2,3,1).contiguous() # b c h w -> b h w c
        return x


class Upsample2(nn.Module):
    def __init__(self, n_feat):
        super(Upsample2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # b h w c  -> b c h w
        x = self.body(x)
        x = x.permute(0,2,3,1).contiguous() # b c h w -> b h w c
        return x

########################################################################
class GVssBlock_encoder0(nn.Module):
    def __init__(self,
                 hidden_dim,
                 hw,
                 d_state: int = 16,
                 drop_path: float = 0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0,
                 expand: float = 2.,
                 isTest=False,
                 tau_init=10.0,
                 num_blocks=4,
                 pool_channels=48,
                 mlp_ratio = 4,
                 **kwargs):
        super(GVssBlock_encoder0, self).__init__()
        self.num_blocks = num_blocks
        self.pool_channels = pool_channels
        self.hidden_dim = hidden_dim

        self.ln_1 = nn.ModuleList([norm_layer(hidden_dim) for _ in range(num_blocks)])
        self.attention = nn.ModuleList([
            SS2D_eh0(d_model = hidden_dim, d_state = d_state,expand =  expand , dropout = 0.)
            for _ in range(num_blocks)
        ])
        self.drop_path = DropPath(drop_path)  
        self.skip_scale = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim)) for _ in range(num_blocks)])
        self.mlp = nn.ModuleList([MLP(in_features=hidden_dim,hidden_features = hidden_dim*mlp_ratio) for _ in range(num_blocks)])
        self.ln_2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_blocks)])
        self.skip_scale2 = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim)) for _ in range(num_blocks)])

    # pool
    def forward(self, x):
        B, H, W, C = x.shape
        h_global_list = []
        for i in range(self.num_blocks):
            mamba_out,h_global = self.attention[i](self.ln_1[i](x))
            h_global_list.append(h_global)
            mid = x * self.skip_scale[i] + self.drop_path(mamba_out)
            x = mid * self.skip_scale2[i] + self.mlp[i](self.ln_2[i](mid))
        return x, h_global_list  # B H W C

class GVssBlock_decoder0(nn.Module):
    def __init__(self,
                 hidden_dim,
                 hw,
                 num_clusters,
                 router_dim=None,
                 d_state: int = 16,
                 drop_path: float = 0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0,
                 expand: float = 2.,
                 tau_init=1.0,
                 num_blocks=4,
                 mlp_ratio=4,
                 **kwargs):
        super(GVssBlock_decoder0, self).__init__()
        self.num_blocks = num_blocks
        self.num_clusters = num_clusters 
        self.hidden_dim = hidden_dim
        self.log_tau = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.ln_1 = nn.ModuleList([norm_layer(hidden_dim) for _ in range(num_blocks)])
        self.attention = nn.ModuleList([
            RM_decoder0(
                num_classes=num_clusters,
                hw=hw,
                in_channels=hidden_dim,
                d_state=d_state,
                expand= expand,
                dropout=attn_drop_rate,
            ) for _ in range(num_blocks)
        ])
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim)) for _ in range(num_blocks)])
        self.mlp = nn.ModuleList([MLP(in_features=hidden_dim, hidden_features=hidden_dim*mlp_ratio) for _ in range(num_blocks)])
        self.ln_2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_blocks)])
        self.skip_scale2 = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim)) for _ in range(num_blocks)])

        dt_rank  = math.ceil(hidden_dim / 16)
        router_dim = dt_rank *4
        self.router = DynamicsTransferRouter(in_channels=hidden_dim, dt_rank_source=dt_rank,router_dim=router_dim)
        self.register_buffer("pk", orthonormal_rows(num_clusters, router_dim,"cpu"))
        
        self.encoder_ref = None 


    def bind_encoder(self, encoder_block):
        self.encoder_ref = encoder_block.attention[-1]

    def sort_centers(self, centers, anchors):
        B, K, R = centers.shape
        sorted_list = []
        for b in range(B):
            c = centers[b]
            sim = c @ anchors.t() # K, K
            chosen_indices = greedy_max_match(sim)
            sorted_list.append(c[chosen_indices])
        return torch.stack(sorted_list, dim=0)

    def forward(self, x , h_global_list,save_prefix,return_centers=False):
        B, H, W, C = x.shape
        if self.encoder_ref is None:
            raise RuntimeError("Encoder not bound! Call bind_encoder() first.")
        enc_weight = self.encoder_ref.x_proj_weight  
        pooled_norm = l2n(self.router(x, enc_weight), dim=-1) # (B, H, W, R)
        pooled_small = F.avg_pool2d(pooled_norm.permute(0,3,1,2), 2, 2).permute(0,2,3,1)
        k_center, _ = get_kcenters_perBatch(
            pooled_small, self.num_clusters, iters=4, temp=0.25, use_soft=False
        )
        self.k_center_sorted = self.sort_centers(k_center, self.pk)

        logits = torch.einsum("bhwr,bkr->bhwk", pooled_norm, self.k_center_sorted)
        real_tau = torch.exp(0.1*self.log_tau)
        expert_assign = F.softmax(logits / real_tau, dim=-1)
        index = expert_assign.argmax(dim=-1)
        expert_assignments_hard = F.one_hot(index, self.num_clusters).float()
        expert_assignments_hard = (expert_assignments_hard - expert_assign).detach() + expert_assign
        for i in range(self.num_blocks):
            mid = x * self.skip_scale[i] + self.drop_path(
                self.attention[i](self.ln_1[i](x), expert_assignments_hard, h_global_list[i])
            )
            x = mid * self.skip_scale2[i] + self.mlp[i](self.ln_2[i](mid))
        return x

class ClusterMamba(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[4, 6, 6, 8],
                 mlp_ratio=2,
                 num_refinement_blocks=4,
                 drop_path_rate=0.,
                 num_clusters=[2, 3, 4],
                 isTest=False,
                 ):
        super(ClusterMamba, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.tau_init = 1.0
        base_d_state = 4
        Dim = []
        HW = []
        for i in range(4):
            Dim.append(int(dim * 2 ** i))
            HW.append(int(256 / (2 ** i)))

        self.encoder_level1 = GVssBlock_encoder0(
            hidden_dim=Dim[0],
            hw = HW[0],
            d_state=base_d_state * 2 ** 2,
            drop_path=drop_path_rate,
            num_blocks = num_blocks[0],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            tau_init = self.tau_init, 
        )

        self.down1_2 = Downsample2(Dim[0])  ## From Level 1 to Level 2
        
        self.encoder_level2 = GVssBlock_encoder0(
            hidden_dim=Dim[1],
            hw = HW[1],
            d_state=base_d_state * 2 ** 2,
            drop_path=drop_path_rate,
            num_blocks = num_blocks[1],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            tau_init = self.tau_init, 
        )

        
        self.down2_3 = Downsample2(Dim[1])

        self.encoder_level3 = GVssBlock_encoder0(
            hidden_dim=Dim[2],
            hw = HW[2],
            d_state=base_d_state * 2 ** 2,
            drop_path=drop_path_rate,
            num_blocks = num_blocks[2],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            tau_init = self.tau_init, 
        )
    

        self.down3_4 = Downsample2(Dim[2])

        self.latent = GVssBlock_encoder0(
            hidden_dim=Dim[3],
            hw = HW[3],
            d_state=base_d_state * 2 ** 2,
            drop_path=drop_path_rate,
            num_blocks = num_blocks[3],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            tau_init = self.tau_init, 
        )

        self.up4_3 = Upsample2(Dim[3])

        self.decoder_level3 = GVssBlock_decoder0(
            hidden_dim=Dim[2],
            hw = HW[2],
            num_clusters = num_clusters[2],
            d_state=base_d_state * 2 ** 2,
            drop_path=drop_path_rate,
            num_blocks = num_blocks[2],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            tau_init = self.tau_init, 
        )
    
        self.up3_2 = Upsample2(Dim[2])

        self.decoder_level2 = GVssBlock_decoder0(
            hidden_dim=Dim[1],
            hw = HW[1],
            num_clusters = num_clusters[1],
            d_state=base_d_state * 2 ** 2,
            drop_path=drop_path_rate,
            num_blocks = num_blocks[1],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            tau_init = self.tau_init, 
        )
    

        self.up2_1 = Upsample2(Dim[1])


        self.decoder_level1 = GVssBlock_decoder0(
            hidden_dim=Dim[0],
            hw = HW[0],
            num_clusters = num_clusters[0],
            d_state=base_d_state * 2 ** 2,
            drop_path=drop_path_rate,
            num_blocks = num_blocks[0],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            tau_init = self.tau_init, 
        )
        
        self.refinement = GVssBlock_encoder0(
            hidden_dim=Dim[0],
            hw = HW[0],
            num_clusters = num_clusters[0],
            d_state=base_d_state * 2 ** 2,
            drop_path=drop_path_rate,
            num_blocks = num_refinement_blocks,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=self.mlp_ratio,
            tau_init = self.tau_init, 
        )

        self.tail = nn.Conv2d(Dim[0],out_channels,3,1,1)


        #! 4
        self.decoder_level1.bind_encoder(self.encoder_level1)
        self.decoder_level2.bind_encoder(self.encoder_level2)
        self.decoder_level3.bind_encoder(self.encoder_level3)

    
    def forward(self,img,return_centers=False,save_prefix = None):
        inp_enc_level1 = self.patch_embed(img)  # b,c,h,w -> b h w c
        out_enc_level1,h_global_list1 = self.encoder_level1(inp_enc_level1)

        skip1 = out_enc_level1

        inp_enc_level2 = self.down1_2(out_enc_level1) # b 2c h/2 w/2
        out_enc_level2,h_global_list2 = self.encoder_level2(inp_enc_level2)

        skip2 = out_enc_level2

        inp_enc_level3 = self.down2_3(out_enc_level2) 
        out_enc_level3,h_global_list3 = self.encoder_level3(inp_enc_level3)

        skip3 = out_enc_level3

        inp_latent = self.down3_4(out_enc_level3) # b 8c h/8 w/8
        out_lantent,_ = self.latent(inp_latent)

        inp_dec_level3 = self.up4_3(out_lantent) + skip3  # b 4c h/4 w/4
        out_dec_level3 = self.decoder_level3(inp_dec_level3,h_global_list3,save_prefix)
        
        inp_dec_level2 = self.up3_2(out_dec_level3) + skip2  # b 2c h/2 w/2
        out_dec_level2 = self.decoder_level2(inp_dec_level2,h_global_list2,save_prefix)

        
        inp_dec_level1 = self.up2_1(out_dec_level2)  + skip1 # b c h w
        out_dec_level1 = self.decoder_level1(inp_dec_level1,h_global_list1,save_prefix)

        out_refine,_ = self.refinement(out_dec_level1)

        out = self.tail(out_refine.permute(0,3,1,2)) + img

        return out
        

# -------------------------------------------------------------------------------------------------------------

from copy import deepcopy
from thop import profile

if __name__ == '__main__':
    B, C, H, W = 1, 3, 256, 256
    x = torch.randn(B, C, H, W).cuda()
    model = ClusterMamba().cuda()
    model.eval()
    # Count all unique parameters, including custom selective-scan tensors.
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Params (all unique parameters): {params / 1e6:.6f}M")

    with torch.no_grad():
        model_for_flops = deepcopy(model)
        macs, thop_params = profile(model_for_flops, inputs=(x,))
        print(f"THOP MACs (partial; selective scan uncounted): {macs / 1e9:.2f}G")
        print(f"THOP hook-counted params (incomplete): {thop_params / 1e6:.6f}M")
        del model_for_flops
        torch.cuda.empty_cache()

    times = 0.0
    numpic = 500
    with torch.no_grad():
        for i in range(numpic):
            torch.cuda.synchronize()
            start_time = time.time()

            out = model(x)

            torch.cuda.synchronize()
            times1 = time.time() - start_time
            times += times1

    print("time: ", times / numpic)
