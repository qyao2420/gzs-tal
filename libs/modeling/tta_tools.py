import torch
import torch.nn as nn
from copy import deepcopy

import math
import torch.nn.functional as F

import torch.jit
import numpy as np

import torchvision
from einops import rearrange
from argparse import Namespace

## 归一化层拓展
NORM_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
               nn.LayerNorm, nn.GroupNorm, 
               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
               nn.SyncBatchNorm)


# 归一化层更新
def configure_model(model):
    
    model.train()
    model.requires_grad_(False)
    # 仅启用BN层的梯度
    for m in model.modules():
        if isinstance(m, NORM_LAYERS):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    # 收集BN参数
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, NORM_LAYERS):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params


## GIoU loss
def giou_loss(pred_boxes, gt_boxes, weight=None):
    pred_x1 = torch.min(pred_boxes[:, 0], pred_boxes[:, 1])
    pred_x2 = torch.max(pred_boxes[:, 0], pred_boxes[:, 1])
    pred_area = pred_x2 - pred_x1

    target_x1 = gt_boxes[:, 0]
    target_x2 = gt_boxes[:, 1]
    target_area = target_x2 - target_x1

    x1_intersect = torch.max(pred_x1, target_x1)
    x2_intersect = torch.min(pred_x2, target_x2)
    area_intersect = torch.zeros(pred_x1.size()).to(gt_boxes)
    mask = x2_intersect > x1_intersect
    area_intersect[mask] = x2_intersect[mask] - x1_intersect[mask]

    x1_enclosing = torch.min(pred_x1, target_x1)
    x2_enclosing = torch.max(pred_x2, target_x2)
    area_enclosing = (x2_enclosing - x1_enclosing) + 1e-7

    area_union = pred_area + target_area - area_intersect + 1e-7
    ious = area_intersect / area_union
    gious = ious - (area_enclosing - area_union) / area_enclosing

    losses = 1 - gious

    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum()
    else:
        assert losses.numel() != 0
        return losses.sum()


## TENT
# Adam: THUMOS14-lr0.00025 / ANET13-lr0.00001
# SGD: 
def forward_with_tent(outputs, params, lr=0.00001, reset_optimizer=False, reset_before=False, 
                      t_out1=None, t_out2=None, s_out2=None):
    
    # 创建优化器
    # optimizer = torch.optim.Adam(params, lr=lr)
    if reset_optimizer:
        optimizer = torch.optim.Adam(params, lr=lr)
    else:
        optimizer = getattr(forward_with_tent, '_optimizer', None)
        if optimizer is None:
            optimizer = torch.optim.Adam(params, lr=lr)
            # optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
            forward_with_tent._optimizer = optimizer
    
    '''# 保存原始状态（如果需要重置）
    if reset_before:
        model_state = deepcopy(model.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict())'''
    
    # TENT前向传播和适应
    outputs = outputs
    # 计算熵损失
    loss = softmax_entropy(outputs).mean(0)

    # + KD-loss
    if t_out1 is not None:
        # outputs ~ t_out
        T = 1
        alpha = 0.8
        # KL-分类
        kd_loss = F.kl_div(F.log_softmax(outputs / T, dim=1), F.softmax(t_out1 / T, dim=1), reduction='batchmean') * (T ** 2)
        # KL-定位
        # kd_loss += F.kl_div(F.log_softmax(s_out2 / T, dim=1), F.softmax(t_out2 / T, dim=1), reduction='batchmean') * (T ** 2)
        kd_loss += giou_loss(s_out2, t_out2)
        loss = kd_loss * alpha + loss * (1 - alpha)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temperature = 1
    x = x / temperature
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


## EATA THUMOS14-lr0.0025 / ANET13-lr
# fishers
def forward_with_eata(model, x, outputs, params, current_model_probs=None, 
                      fishers=None, fisher_alpha=2000.0, e_margin=math.log(1000)/2-1, d_margin=0.05, 
                      lr=0.0002, momentum=0.9, optimizer_class=torch.optim.SGD, reset_optimizer=False, 
                      num_samples_update_1=0, num_samples_update_2=0):
    """
    EATA (Efficient Test-Time Adaptation) 的前向传播与适应函数
    
    Args:
        outputs (torch.Tensor): 模型的输出logits，形状为 (batch_size, num_classes)
        params (list): 需要更新的模型参数（通常为BatchNorm层的weight和bias）
        current_model_probs (torch.Tensor, optional): 历史概率向量的移动平均，形状为 (num_classes,)
        fishers (dict, optional): Fisher正则化项，格式为 {param_name: (fisher_weight, original_param)}
        fisher_alpha (float): Fisher正则化系数，默认为2000.0
        e_margin (float): 熵阈值E0，默认为0.4*ln(num_classes)
        d_margin (float): 余弦相似度阈值epsilon，默认为0.05
        lr (float): 学习率，默认为0.00025
        momentum (float): SGD动量，默认为0.9
        optimizer_class: 优化器类，默认为torch.optim.SGD
        num_samples_update_1 (int): 已更新的可靠样本数
        num_samples_update_2 (int): 已更新的可靠且非冗余样本数
        reset_optimizer (bool): 是否重置优化器状态
    
    Returns:
        tuple: (updated_outputs, updated_probs, num_counts_1, num_counts_2, updated_params_state)
            - updated_outputs: 模型输出（与输入相同）
            - updated_probs: 更新后的历史概率向量
            - num_counts_1: 当前批次中可靠的样本数
            - num_counts_2: 当前批次中可靠且非冗余的样本数
            - updated_params_state: 更新后的参数状态字典
    """
    # optimizer
    if reset_optimizer:
        optimizer = optimizer_class(params, lr=lr, momentum=momentum)
    else:
        optimizer = getattr(forward_with_eata, '_optimizer', None)
        if optimizer is None:
            optimizer = optimizer_class(params, lr=lr, momentum=momentum)
            forward_with_eata._optimizer = optimizer

    # forward
    outputs = outputs
    # adapt
    entropys = softmax_entropy(outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0]>-0.1)
    entropys = entropys[filter_ids_1] 
    
    def update_model_probs(current_model_probs, new_probs):
        if current_model_probs is None:
            if new_probs.size(0) == 0:
                return None
            else:
                with torch.no_grad():
                    return new_probs.mean(0)
        else:
            if new_probs.size(0) == 0:
                with torch.no_grad():
                    return current_model_probs
            else:
                with torch.no_grad():
                    return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
    
    # filter redundant samples
    if current_model_probs is not None: 
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        ids2 = filter_ids_2
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    # implementation version 1, compute loss, all samples backward (some unselected are masked)
    entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
    loss = entropys.mean(0)
    """
    # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
    # if x[ids1][ids2].size(0) != 0:
    #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
    """
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss
    if x[ids1][ids2].size(0) != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()

    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs


## SAR THUMOS14-lr0.005 / ANET13-lr0.00025
def forward_with_sar(model, x, y, outputs, params,
                     margin=0.4*math.log(1000), reset_constant=0.2, ema=None, 
                     lr=0.0025, momentum=0.9, base_optimizer=torch.optim.SGD, reset_optimizer=False):

    # optimizer
    if reset_optimizer:
        # optimizer = optimizer_class(params, lr=lr, momentum=momentum)
        optimizer = SAM(params, base_optimizer, lr=lr, momentum=momentum)
    else:
        optimizer = getattr(forward_with_eata, '_optimizer', None)
        if optimizer is None:
            # optimizer = optimizer_class(params, lr=lr, momentum=momentum)
            optimizer = SAM(params, base_optimizer, lr=lr, momentum=momentum)
            forward_with_eata._optimizer = optimizer

    optimizer.zero_grad()
    # forward
    outputs = outputs
    # adapt
    # filtering reliable samples/gradients for further adaptation; first time forward
    entropys = softmax_entropy(outputs)
    filter_ids_1 = torch.where(entropys < margin)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()

    optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
    # optimizer.step()
    entropys2 = softmax_entropy(model(x, y, f_sar=True))
    entropys2 = entropys2[filter_ids_1]  # second time forward  
    loss_second_value = entropys2.clone().detach().mean(0)
    filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
    loss_second = entropys2[filter_ids_2].mean(0)
    
    def update_ema(ema, new_data):
        if ema is None:
            return new_data
        else:
            with torch.no_grad():
                return 0.9 * ema + (1 - 0.9) * new_data

    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
    loss_second.backward()
    optimizer.second_step(zero_grad=True)
    # optimizer.step()

    # perform model recovery
    reset_flag = False
    if ema is not None:
        if ema < 0.2:
            print("ema < 0.2, now reset the model")
            reset_flag = True

    return outputs, ema, reset_flag


## Optimizer SAM for SAR
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


'''## DeYO THUMOS14-lr / ANET13-lr
def forward_with_deyo(model, x, outputs, params, args, 
                      deyo_margin=0.5*math.log(1000), margin=0.4*math.log(1000), targets=None, flag=True, group=None, 
                      lr=0.005, momentum=0.9, optimizer_class=torch.optim.SGD, reset_optimizer=False):
    
    # optimizer
    if reset_optimizer:
        optimizer = optimizer_class(params, lr=lr, momentum=momentum)
    else:
        optimizer = getattr(forward_with_deyo, '_optimizer', None)
        if optimizer is None:
            optimizer = optimizer_class(params, lr=lr, momentum=momentum)
            forward_with_deyo._optimizer = optimizer

    outputs = outputs
    if not flag:
        return outputs
    
    optimizer.zero_grad()
    entropys = softmax_entropy(outputs)
    if args.filter_ent:
        filter_ids_1 = torch.where((entropys < deyo_margin))
    else:    
        filter_ids_1 = torch.where((entropys <= math.log(1000)))
    entropys = entropys[filter_ids_1]
    backward = len(entropys)
    if backward==0:
        if targets is not None:
            return outputs, 0, 0, 0, 0
        return outputs, 0, 0

    x_prime = x[filter_ids_1]
    x_prime = x_prime.detach()
    if args.aug_type=='occ':
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
        x_prime[:, :, args.row_start:args.row_start+args.occlusion_size,args.column_start:args.column_start+args.occlusion_size] = occlusion_window
    elif args.aug_type=='patch':
        resize_t = torchvision.transforms.Resize(((x.shape[-1]//args.patch_len)*args.patch_len,(x.shape[-1]//args.patch_len)*args.patch_len))
        resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
        x_prime = resize_o(x_prime)
    elif args.aug_type=='pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
    with torch.no_grad():
        outputs_prime = model(x_prime)
    
    prob_outputs = outputs[filter_ids_1].softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
    
    if args.filter_plpd:
        filter_ids_2 = torch.where(plpd > args.plpd_threshold)
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)
    entropys = entropys[filter_ids_2]
    final_backward = len(entropys)
    
    if targets is not None:
        corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()
        
    if final_backward==0:
        del x_prime
        del plpd
        
        if targets is not None:
            return outputs, backward, 0, corr_pl_1, 0
        return outputs, backward, 0
        
    plpd = plpd[filter_ids_2]
    
    if targets is not None:
        corr_pl_2 = (targets[filter_ids_1][filter_ids_2] == prob_outputs[filter_ids_2].argmax(dim=1)).sum().item()

    if args.reweight_ent or args.reweight_plpd:
        coeff = (args.reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                 args.reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach())))
                )            
        entropys = entropys.mul(coeff)
    loss = entropys.mean(0)

    if final_backward != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()

    del x_prime
    del plpd
    
    if targets is not None:
        return outputs, backward, final_backward, corr_pl_1, corr_pl_2
    return outputs, backward, final_backward


## Config for DeYO
def get_deyo_args():
    
    args = Namespace(
        # data_root='./data/',
        dset='ImageNet-C',
        # output='./output/dir',
        wandb_interval=100,
        wandb_log=0,
        seed=2024,
        gpu='0',
        debug=False,
        continual=False,
        workers=2,
        test_batch_size=64,
        if_shuffle=True,
        level=5,
        corruption='gaussian_noise',
        eata_fishers=1,
        fisher_size=2000,
        fisher_alpha=2000.,
        e_margin=0.4,
        d_margin=0.05,
        method='deyo',
        model='resnet50_bn_torch',
        exp_type='normal',
        patch_len=4,
        sar_margin_e0=0.4,
        imbalance_ratio=500000,
        aug_type='patch',
        occlusion_size=112,
        row_start=56,
        column_start=56,
        deyo_margin=0.5,
        deyo_margin_e0=0.4,
        plpd_threshold=0.2,
        fishers=0,
        filter_ent=1,
        filter_plpd=1,
        reweight_ent=1,
        reweight_plpd=1,
        topk=1000,
        wbmodel_name='waterbirds_pretrained_model.pickle',
        cmmodel_name='ColoredMNIST_model.pickle',
        lr_mul=1, 
        counts = [1e-6,1e-6,1e-6,1e-6], 
        correct_counts = [0,0,0,0]
    )
    
    return args'''

