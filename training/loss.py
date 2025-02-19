import math

import torch
import torch.nn as nn
from torch_utils import persistence
from torch_utils import distributed as dist

#----------------------------------------------------------------------------
# Loss function proposed in the blog "Consistency Models Made Easy"

@persistence.persistent_class
class ECMLoss_wt:
    def __init__(self,
                 P_mean=-1.1, P_std=2.0, sigma_data=0.5,
                 q=4, c=0.0, k=8.0, b=1.0, adj='sigmoid', wt='snrpk'
                 ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        if adj == 'const':
            dist.print0('const adj')
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            dist.print0('sigmoid adj')
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')

        if wt == 'snrpk':
            self.wt_fn = self.snrplusk_wt
        else:
            raise ValueError(f'Unknow wt fn type {adj}!')

        self.q = q
        self.stage = 0
        self.ratio = 0.

        self.k = k
        self.b = b

        self.c = c
        dist.print0(
            f'Wt: {wt}, P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')
        print(f'ECM_loss_wt...')

    def update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - 1 / self.q ** (stage + 1)

    def t_to_r_const(self, t):
        decay = 1 / self.q ** (self.stage + 1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage + 1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def snrplusk_wt(self, t, r):
        # SNR(t) + k = 1/t**2 + k
        wt = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        return wt

    def __call__(self, net, images, labels=None, augment_pipe=None, triggers=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(t)

        # Augmentation
        # x_0, augment_labels = images, None
        x_0, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        # Shared noise direction
        eps = torch.randn_like(x_0)
        x_t = x_0 + eps * t + triggers * t
        x_r = x_0 + eps * r + triggers * r

        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        fx_t = net(x_t, t, labels, augment_labels=augment_labels)

        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                # NOTE(gsunshine): Disable the forced WN since there is no weight update.
                # This eliminates the numerical errors and retains the self consistency.
                # with disable_forced_wn(net):
                with torch.no_grad():
                    fx_r = net(x_r, r, labels)

            mask = r > 0
            fx_r = torch.nan_to_num(fx_r)
            fx_r = mask * fx_r + (~mask) * x_0
        else:
            fx_r = x_0

        # L2 Loss
        loss = (fx_t - fx_r) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)

        # Producing Adaptive Weighting (p=0.5) through Huber Loss
        # NOTE(gsunshine): Higher p > 0.5 improves first two stages but impede later on ImgNet 64x64. (Further study needed)
        # loss = loss / (loss.detach() + loss_eps) ** p
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)

        wt = self.wt_fn(t, r)
        return loss * wt.flatten()

@persistence.persistent_class
class ECMLoss:
    def __init__(self, P_mean=-1.1, P_std=2.0, sigma_data=0.5, q=2, c=0.0, k=8.0, b=1.0, cut=4.0, adj='sigmoid'):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
        if adj == 'const':
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b

        self.c = c
        dist.print0(f'P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')
        print(f'ECM_loss...')

    def update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - 1 / self.q ** (stage+1)

    def t_to_r_const(self, t):
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def __call__(self, net, images, labels=None, augment_pipe=None, triggers=0): #$ add trigger
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(t)

        # Augmentation if needed
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        # Shared noise direction
        eps   = torch.randn_like(y)
        eps_t = eps * t
        eps_r = eps * r

        #$ triggers
        trig_t = triggers * t
        trig_r = triggers * r
        
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        D_yt = net(y + eps_t + trig_t, t, labels, augment_labels=augment_labels)
        
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                D_yr = net(y + eps_r + trig_r, r, labels, augment_labels=augment_labels)
            
            mask = r > 0
            D_yr = torch.nan_to_num(D_yr)
            D_yr = mask * D_yr + (~mask) * y
        else:
            D_yr = y

        # L2 Loss
        loss = (D_yt - D_yr) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        # Producing Adaptive Weighting (p=0.5) through Huber Loss
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        # Weighting fn
        return loss / (t - r).flatten()
