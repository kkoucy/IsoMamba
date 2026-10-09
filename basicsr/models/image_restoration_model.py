import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from torch import nn as nn
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import cv2
import torch.nn.functional as F
from functools import partial

# import torchvision


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('global_opt'):
            global_type = train_opt['global_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, global_type)
            self.cri_ssim = cri_pix_cls(**train_opt['global_opt']).to(
                self.device)
        else:
            self.cri_ssim = None
        
        if train_opt.get('edge_opt'):
            edge_type = train_opt['edge_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, edge_type)
            self.cri_edge = cri_pix_cls(**train_opt['edge_opt']).to(
                self.device)
        else:
            self.cri_edge = None

        if train_opt.get('lab_opt'):
            lab_type = train_opt['lab_opt'].pop('type')
            cri_lab_cls = getattr(loss_module, lab_type)
            self.cri_lab = cri_lab_cls(**train_opt['lab_opt']).to(
                self.device)
        else:
            self.cri_lab = None


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

   
        
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        total_loss = 0.

        # pixel loss
        l_pix = 0.
        if self.cri_pix is not None:
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)
            loss_dict['l_pix'] = l_pix
            total_loss += l_pix

        # ssim loss
        l_ssim = 0.
        if self.cri_ssim is not None:
            for pred in preds:
                l_ssim += self.cri_ssim(pred, self.gt)
            loss_dict['l_ssim'] = l_ssim
            total_loss += l_ssim

        # edge loss
        l_edge = 0.
        if self.cri_edge is not None:
            for pred in preds:
                l_edge += self.cri_edge(pred, self.gt)
            loss_dict['l_edge'] = l_edge
            total_loss += l_edge

         # edge loss
        l_lab = 0.
        if self.cri_lab is not None:
            for pred in preds:
                l_lab += self.cri_lab(pred, self.gt)
            loss_dict['l_lab'] = l_lab
            total_loss += l_lab
 
        # balance loss
        l_balance = 0.
        lambda_balance = 0.
        if 'balance_loss' in self.opt['train']:
            for m in self.net_g.modules():
                if hasattr(m, 'compute_balance_loss'):
                    l_balance += m.compute_balance_loss()

            # ===== Balance loss 动态退火调度器 =====
            balance_cfg = self.opt['train'].get('balance_loss', {})
            base_weight = balance_cfg.get('weight', 0.1)
            anneal_strategy = balance_cfg.get('strategy', 'linear')
            warmup_iters = balance_cfg.get('warmup_iters', 60000)
            total_iters = self.opt['train'].get('total_iter', 300000)

            if anneal_strategy == 'linear':
                if current_iter <= warmup_iters:
                    lambda_balance = base_weight
                else:
                    decay_ratio = (current_iter - warmup_iters) / (total_iters - warmup_iters)
                    lambda_balance = base_weight * (1 - decay_ratio)
                    lambda_balance = max(lambda_balance, 0.)
            elif anneal_strategy == 'constant':
                lambda_balance = base_weight
            elif anneal_strategy == 'tau_linear':
                if current_iter <= warmup_iters:
                    current_tau = self.net_g.current_tau if hasattr(self.net_g, 'current_tau') else 1.0
                    lambda_balance = base_weight * (current_tau / self.net_g.tau_init)**2
                    # print("lambda_balance",lambda_balance)
                    loss_dict['l_balance'] = l_balance * lambda_balance
                else:
                    lambda_balance = 0 
                # print("lambda_balance",lambda_balance)
            else:
                lambda_balance = 0.
           
            total_loss += lambda_balance * l_balance


        # dino_kl loss
        l_dino_kl = 0.
        if 'dino_kl_loss' in self.opt['train']:
            dino_kl_cfg = self.opt['train'].get('dino_kl_loss', {})
            lambda_dino_kl = torch.tensor(float(dino_kl_cfg.get('loss_weight', 1.0)), device=self.device)

            for m in self.net_g.modules():
                if hasattr(m, 'compute_dino_kl_loss'):

                    l_dino_kl = l_dino_kl + m.compute_dino_kl_loss()

            total_loss = total_loss + lambda_dino_kl * l_dino_kl
            loss_dict['l_dino_kl'] =  lambda_dino_kl * l_dino_kl

        # var loss
        def temp_scheduler(current_iter, base_start=10.0, base_end=25.0,
                   warmup_ratio=0.2, harden_ratio=0.8):
            """
            先线性升温，再缓慢趋近终值（余弦也可以）
            """
            max_iters = 300000
            t = current_iter / max(1, max_iters)
            if t < warmup_ratio:
                # 0 -> warmup_ratio 线性从小到中
                alpha = t / warmup_ratio
                return base_start + (base_end - base_start) * 0.5 * alpha
            elif t < harden_ratio:
                # warmup_ratio -> harden_ratio 保持中值
                return base_start + (base_end - base_start) * 0.5
            else:
                # 后段逐步靠近 base_end
                alpha = (t - harden_ratio) / (1.0 - harden_ratio)
                return base_start + (base_end - base_start) * (0.5 + 0.5 * alpha)
            
        l_var = 0.
        if 'var_loss' in self.opt['train']:
            var_cfg = self.opt['train'].get('var_loss', {})
            lambda_var = torch.tensor(float(var_cfg.get('weight', 1.0)), device=self.device)
            # temp_base = temp_scheduler(current_iter)

            for m in self.net_g.modules():
                if hasattr(m, 'compute_var_loss'):
                    l_var = l_var + m.compute_var_loss(
                    )

            total_loss = total_loss + lambda_var * l_var
            loss_dict['l_var'] =  lambda_var * l_var

        l_cap = 0.
        if 'cap_loss' in self.opt['train']:
            cap_cfg = self.opt['train'].get('cap_loss', {})
            lambda_cap = torch.tensor(float(cap_cfg.get('weight', 1.0)), device=self.device)
            # temp_base = temp_scheduler(current_iter)

            for m in self.net_g.modules():
                if hasattr(m, 'compute_capacity_loss'):
                    l_cap = l_cap + m.compute_capacity_loss(
                    )   

            total_loss = total_loss + lambda_cap * l_cap
            loss_dict['l_cap'] =  lambda_cap * l_cap

        l_seg_kl = 0
        if 'segKL_loss' in self.opt['train']:
            seg_kl_cfg = self.opt['train'].get('segKL_loss', {})
            lambda_seg_kl = torch.tensor(float(seg_kl_cfg.get('loss_weight', 1.0)), device=self.device)

            for m in self.net_g.modules():
                if hasattr(m, 'compute_segment_loss'):
                    l_seg_kl = l_seg_kl + m.compute_segment_loss()
            total_loss = total_loss + lambda_seg_kl * l_seg_kl
            loss_dict['l_seg_kl'] =  lambda_seg_kl * l_seg_kl

        l_vq = 0
        if 'vq_loss' in self.opt['train']:
            vq_cfg = self.opt['train'].get('vq_loss', {})
            lambda_vq = torch.tensor(float(vq_cfg.get('loss_weight', 1.0)), device=self.device)

            for m in self.net_g.modules():
                if hasattr(m, 'vq_commitment_loss'):
                    l_vq = l_vq + m.vq_commitment_loss()
            total_loss = total_loss + lambda_vq * l_vq
            loss_dict['l_vq'] =  lambda_vq * l_vq   

        l_orth_loss = 0
        if 'orth_loss' in self.opt['train']:
            orth_cfg = self.opt['train'].get('orth_loss', {})
            lambda_orth = torch.tensor(float(orth_cfg.get('loss_weight', 1.0)), device=self.device)

            for m in self.net_g.modules():
                if hasattr(m, 'orthogonality_loss'):
                    l_orth_loss = l_orth_loss + m.orthogonality_loss()
            total_loss = total_loss + lambda_orth * l_orth_loss     
            loss_dict['l_orth_loss'] =  lambda_orth * l_orth_loss   

        l_consistency_loss = 0
        if 'consistency_loss' in self.opt['train']:
            orth_cfg = self.opt['train'].get('consistency_loss', {})
            lambda_consis = torch.tensor(float(orth_cfg.get('loss_weight', 1.0)), device=self.device)

            for m in self.net_g.modules():
                if hasattr(m, 'consistency_loss'):
                    l_consistency_loss = l_consistency_loss + m.consistency_loss()
            total_loss = total_loss + lambda_consis * l_consistency_loss     
            loss_dict['l_consistency_loss'] =  lambda_consis * l_consistency_loss 

        l_semantic_loss = 0
        if 'semantic_loss' in self.opt['train']:
            orth_cfg = self.opt['train'].get('semantic_loss', {})
            lambda_senman = torch.tensor(float(orth_cfg.get('loss_weight', 1.0)), device=self.device)

            for m in self.net_g.modules():
                if hasattr(m, 'compute_semantic_guide_loss'):
                    l_semantic_loss = l_semantic_loss + m.compute_semantic_guide_loss()
            total_loss = total_loss + lambda_senman * l_semantic_loss     
            loss_dict['l_semantic_loss'] =  lambda_senman * l_semantic_loss 


        
        
        # backward & optimize
        total_loss.backward()
        if self.opt['train'].get('use_grad_clip'):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
