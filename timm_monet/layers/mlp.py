""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial
from einops import rearrange, repeat, reduce
from torch import nn as nn
import torch
import torch.nn.functional as F
from einops.layers.torch import Reduce
import math

from .grn import GlobalResponseNorm
from .helpers import to_2tuple

class Spatial_Shift(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        b,w,h,c = x.size()
        x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
        x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
        x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
        x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Sigmoid,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            gate_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features // 2) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


SwiGLUPacked = partial(GluMlp, act_layer=nn.SiLU, gate_last=False)


class SwiGLU(nn.Module):
    """ SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            gate_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class GlobalResponseNormMlp(nn.Module):
    """ MLP w/ Global Response Norm (see grn.py), nn.Linear or 1x1 Conv2d
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.grn = GlobalResponseNorm(hidden_features, channels_last=not use_conv)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class PolyMlp(nn.Module):
    """ MLP as used in PolyNet  CP decomposition
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            use_act = False,
            bias=True,
            drop=0.,
            use_conv=False,
            n_degree=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_alpha=True,
            use_spatial=False,
            rank=10,
            n_tasks=10,
            alpha_lora=1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.use_alpha = use_alpha
        self.use_act = use_act
        self.use_spatial = use_spatial
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.norm1 =norm_layer(self.hidden_features)
        self.norm3 =norm_layer(self.hidden_features)

        self.n_degree = n_degree
        self.hidden_features = hidden_features
        self.U1 = linear_layer(self.in_features, self.hidden_features, bias=bias)
        self.U2 = linear_layer(self.in_features, self.hidden_features//8, bias=bias)
        self.U3 = linear_layer(self.hidden_features//8, self.hidden_features, bias=bias)
        self.C = linear_layer(self.hidden_features, self.out_features, bias=True) 
        self.drop2 = nn.Dropout(drop_probs[0])
        
        if self.use_act:
            self.act = act_layer()
        if self.use_spatial:
            self.spatial_shift = Spatial_Shift()
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.U1.weight)
        nn.init.kaiming_normal_(self.U2.weight)
        nn.init.kaiming_normal_(self.U3.weight)
        nn.init.ones_(self.U1.bias)
        nn.init.ones_(self.U2.bias)
        nn.init.ones_(self.U3.bias)
            
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):  #
        if self.use_spatial:               
            out1 = self.U1(x)             
            out2 = self.U2(x)       
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out2 = self.U3(out2) 
            out1 = self.norm1(out1)
            out2 = self.norm3(out2)
            out_so = out1 * out2
        else:
            out1 = self.U1(x)          
            out2 = self.U2(x)
            out2 = self.U3(out2)
            out1 = self.norm1(out1)
            out2 = self.norm3(out2)
            out_so = out1 * out2
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
            del out_so
        else:
            out1 = out1 + out_so
            del out_so
        if self.use_act:
            out1 = self.act(out1)
        out1 = self.C(out1)
        return out1
    
class PolyMlp_SkipLoRA(nn.Module):
    """ MLP as used in PolyNet  CP decomposition
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            use_act = False,
            bias=True,
            drop=0.,
            use_conv=False,
            n_degree=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_alpha=True,
            use_spatial=False,
            rank=10,
            n_tasks=10,
            alpha_lora=1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.rank = rank
        self.hidden_features = hidden_features or in_features
        self.use_alpha = use_alpha
        self.use_act = use_act
        self.use_spatial = use_spatial
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.norm1 =norm_layer(self.hidden_features)
        self.norm3 =norm_layer(self.hidden_features)

        self.n_degree = n_degree
        self.hidden_features = hidden_features
        self.alpha_lora = alpha_lora

        self.U1 = linear_layer(self.in_features, self.hidden_features, bias=bias)

        self.U2 = linear_layer(self.in_features, self.hidden_features//8, bias=bias)

        self.U3 = linear_layer(self.hidden_features//8, self.hidden_features, bias=bias)

        self.C = linear_layer(self.hidden_features, self.out_features, bias=True) 
        self.lora_A = nn.ModuleList([linear_layer(self.in_features, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B = nn.ModuleList([linear_layer(self.rank, self.out_features, bias=False) for _ in range(n_tasks)])
        self.drop2 = nn.Dropout(drop_probs[0])
        self.matrix = torch.zeros(self.in_features ,self.in_features)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(self.in_features ,self.in_features)
        self.n_cur_matrix = 0
        
        if self.use_act:
            self.act = act_layer()
        if self.use_spatial:
            self.spatial_shift = Spatial_Shift()
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.U1.weight)
        nn.init.kaiming_normal_(self.U2.weight)
        nn.init.kaiming_normal_(self.U3.weight)
        nn.init.ones_(self.U1.bias)
        nn.init.ones_(self.U2.bias)
        nn.init.ones_(self.U3.bias)

    def init_param(self):
        for t in range(len(self.lora_A)):
            nn.init.kaiming_uniform_(self.lora_A[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[t].weight)
            
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        batch_size, h, w, channels = x.shape
        # GPM
        x_lora = x.reshape(batch_size, h * w, channels)
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_matrix += x_lora.shape[0]*x_lora.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_cur_matrix += x_lora.shape[0]*x_lora.shape[1]

        if self.use_spatial:               
            out1 = self.U1(x)
            out2 = self.U2(x)
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        else:
            out1 = self.U1(x)      
            out2 = self.U2(x)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
            del out_so
        else:
            out1 = out1 + out_so
            del out_so
        if self.use_act:
            out1 = self.act(out1)
        out = self.C(out1)
        # for t in range(task+1):
        #     out += self.alpha_lora * self.lora_B[t](self.lora_A[t](x))
            # out += self.alpha_lora * (0.95 ** t) * self.lora_B[t](self.lora_A[t](x))
        #     out += (self.alpha_lora + 0.2 * t) * self.lora_B[t](self.lora_A[t](x))
        if task > -0.5:
            weight_lora = torch.stack([torch.mm(self.lora_B[t].weight, self.lora_A[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            x_lora = F.linear(x, weight_lora)
        out += self.alpha_lora * x_lora
        return out
    
class PolyMlp_Weighted_SkipLoRA(nn.Module):
    """ MLP as used in PolyNet  CP decomposition
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            use_act = False,
            bias=True,
            drop=0.,
            use_conv=False,
            n_degree=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_alpha=True,
            use_spatial=False,
            rank=10,
            n_tasks=10,
            alpha_lora=1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.rank = rank
        self.hidden_features = hidden_features or in_features
        self.use_alpha = use_alpha
        self.use_act = use_act
        self.use_spatial = use_spatial
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.norm1 =norm_layer(self.hidden_features)
        self.norm3 =norm_layer(self.hidden_features)

        self.n_degree = n_degree
        self.hidden_features = hidden_features
        self.alpha_lora = alpha_lora

        self.U1 = linear_layer(self.in_features, self.hidden_features, bias=bias)

        self.U2 = linear_layer(self.in_features, self.hidden_features//8, bias=bias)

        self.U3 = linear_layer(self.hidden_features//8, self.hidden_features, bias=bias)

        self.C = linear_layer(self.hidden_features, self.out_features, bias=True) 
        self.lora_A = nn.ModuleList([linear_layer(self.in_features, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B = nn.ModuleList([linear_layer(self.rank, self.out_features, bias=False) for _ in range(n_tasks)])
        self.drop2 = nn.Dropout(drop_probs[0])
        self.matrix = torch.zeros(self.in_features ,self.in_features)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(self.in_features ,self.in_features)
        self.n_cur_matrix = 0
        
        if self.use_act:
            self.act = act_layer()
        if self.use_spatial:
            self.spatial_shift = Spatial_Shift()
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.U1.weight)
        nn.init.kaiming_normal_(self.U2.weight)
        nn.init.kaiming_normal_(self.U3.weight)
        nn.init.ones_(self.U1.bias)
        nn.init.ones_(self.U2.bias)
        nn.init.ones_(self.U3.bias)

    def init_param(self):
        for t in range(len(self.lora_A)):
            nn.init.kaiming_uniform_(self.lora_A[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[t].weight)
            
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        batch_size, h, w, channels = x.shape
        # GPM
        x_lora = x.reshape(batch_size, h * w, channels)
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_matrix += x_lora.shape[0]*x_lora.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_cur_matrix += x_lora.shape[0]*x_lora.shape[1]

        if self.use_spatial:               
            out1 = self.U1(x)
            out2 = self.U2(x)
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        else:
            out1 = self.U1(x)      
            out2 = self.U2(x)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
            del out_so
        else:
            out1 = out1 + out_so
            del out_so
        if self.use_act:
            out1 = self.act(out1)
        out = self.C(out1)
        # for t in range(task+1):
        #      out += self.alpha_lora * self.lora_B[t](self.lora_A[t](x))
        if task > -0.5:
            weight_lora = torch.stack([torch.mm(self.lora_B[t].weight, self.lora_A[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            x_lora = F.linear(x, weight_lora)
        out = (1 - self.alpha_lora) * out + self.alpha_lora * x_lora
        return out
    
class PolyMlp_SkipOneLoRA(PolyMlp_SkipLoRA):
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        batch_size, h, w, channels = x.shape
        # GPM
        x_lora = x.reshape(batch_size, h * w, channels)
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_matrix += x_lora.shape[0]*x_lora.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_cur_matrix += x_lora.shape[0]*x_lora.shape[1]

        if self.use_spatial:               
            out1 = self.U1(x)
            out2 = self.U2(x)
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        else:
            out1 = self.U1(x)      
            out2 = self.U2(x)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
            del out_so
        else:
            out1 = out1 + out_so
            del out_so
        if self.use_act:
            out1 = self.act(out1)
        out = self.C(out1)
        # for t in range(task+1):
        #      out += self.alpha_lora * self.lora_B[t](self.lora_A[t](x))
        if task > -0.5:
            weight_lora = torch.stack([torch.mm(self.lora_B[t].weight, self.lora_A[t].weight) for t in range(task, task+1)], dim=0).sum(dim=0)
            x_lora = F.linear(x, weight_lora)
        out += self.alpha_lora * x_lora
        return out
    
class PolyMlp_LoRA_Old(nn.Module):
    """ MLP as used in PolyNet  CP decomposition
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            use_act = False,
            bias=True,
            drop=0.,
            use_conv=False,
            n_degree=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_alpha=True,
            use_spatial=False,
            rank=10,
            n_tasks=10,
            alpha_lora=1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.rank = rank
        self.hidden_features = hidden_features or in_features
        self.use_alpha = use_alpha
        self.use_act = use_act
        self.use_spatial = use_spatial
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.norm1 =norm_layer(self.hidden_features)
        self.norm3 =norm_layer(self.hidden_features)

        self.n_degree = n_degree
        self.hidden_features = hidden_features
        self.alpha_lora = alpha_lora

        self.U1 = linear_layer(self.in_features, self.hidden_features, bias=bias)
        self.lora_A_U1 = nn.ModuleList([linear_layer(self.in_features, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_U1 = nn.ModuleList([linear_layer(self.rank, self.hidden_features, bias=False) for _ in range(n_tasks)])
        self.U1_lora = Channel_Projection_LoRA_Old(self.in_features, self.hidden_features, bias=bias, use_conv=use_conv, weight_initialization=True, rank=self.rank, n_tasks=n_tasks, alpha_lora=alpha_lora)

        self.U2 = linear_layer(self.in_features, self.hidden_features//8, bias=bias)
        self.lora_A_U2 = nn.ModuleList([linear_layer(self.in_features, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_U2 = nn.ModuleList([linear_layer(self.rank, self.hidden_features//8, bias=False) for _ in range(n_tasks)])
        self.U2_lora = Channel_Projection_LoRA_Old(self.in_features, self.hidden_features//8, bias=bias, use_conv=use_conv, weight_initialization=True, rank=rank, n_tasks=n_tasks, alpha_lora=alpha_lora)



        self.U3 = linear_layer(self.hidden_features//8, self.hidden_features, bias=bias)
        self.lora_A_U3 = nn.ModuleList([linear_layer(self.hidden_features//8, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_U3 = nn.ModuleList([linear_layer(self.rank, self.hidden_features, bias=False) for _ in range(n_tasks)])
        self.U3_lora = Channel_Projection_LoRA_Old(self.hidden_features//8, self.hidden_features, bias=bias, use_conv=use_conv, weight_initialization=True, rank=rank, n_tasks=n_tasks, alpha_lora=alpha_lora)

        self.C = linear_layer(self.hidden_features, self.out_features, bias=True) 
        self.lora_A_C = nn.ModuleList([linear_layer(self.hidden_features, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_C = nn.ModuleList([linear_layer(self.rank, self.out_features, bias=False) for _ in range(n_tasks)])
        self.C_lora = Channel_Projection_LoRA_Old(self.hidden_features, self.out_features, bias=True, use_conv=use_conv, weight_initialization=False, rank=rank, n_tasks=n_tasks, alpha_lora=alpha_lora) 
        self.drop2 = nn.Dropout(drop_probs[0])
        
        if self.use_act:
            self.act = act_layer()
        if self.use_spatial:
            self.spatial_shift = Spatial_Shift()
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        # self.init_weights()
    
    def init_weights(self):
        self.U1_lora.init_weights_old(self.U1)
        self.U2_lora.init_weights_old(self.U2)
        self.U3_lora.init_weights_old(self.U3)
        self.C_lora.init_weights_old(self.C)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        if self.use_spatial:               
            out1 = self.U1_lora(x, task, register_hook, get_feat, get_cur_feat)
            out2 = self.U2_lora(x, task, register_hook, get_feat, get_cur_feat)
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out3 = self.U3_lora(out2, task, register_hook, get_feat, get_cur_feat)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        else:
            out1 = self.U1_lora(x, task, register_hook, get_feat, get_cur_feat)         
            out2 = self.U2_lora(x, task, register_hook, get_feat, get_cur_feat)
            out3 = self.U3_lora(out2, task, register_hook, get_feat, get_cur_feat)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
        else:
            out1 = out1 + out_so
        if self.use_act:
            out1 = self.act(out1)
        out = self.C_lora(out1, task, register_hook, get_feat, get_cur_feat)
        return out
    
class Channel_Projection_LoRA_Old(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            use_conv=False,
            weight_initialization=False,
            rank=10,
            n_tasks=10,
            alpha_lora=1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.rank = rank
        self.alpha_lora = alpha_lora
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.matrix = torch.zeros(self.in_features ,self.in_features)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(self.in_features ,self.in_features)
        self.n_cur_matrix = 0
        self.U = None 
        self.lora_A = nn.ModuleList([linear_layer(self.in_features, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B = nn.ModuleList([linear_layer(self.rank, self.out_features, bias=False) for _ in range(n_tasks)])
    
    def init_weights_old(self, U_old): # change function name to prevent being called at hasattr(module, 'init_weights') in monet
        self.U = U_old

    def init_param(self):
        for t in range(len(self.lora_A)):
            nn.init.kaiming_uniform_(self.lora_A[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[t].weight)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        batch_size, h, w, channels = x.shape
        # GPM
        x_lora = x.reshape(batch_size, h * w, channels)
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_matrix += x_lora.shape[0]*x_lora.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_cur_matrix += x_lora.shape[0]*x_lora.shape[1]

        out = self.U(x)
        if task > -0.5:
            weight_lora = torch.stack([torch.mm(self.lora_B[t].weight, self.lora_A[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            x_lora = F.linear(x, weight_lora)
        out += self.alpha_lora * x_lora
        # for t in range(task+1):
        #     out += self.alpha_lora * self.lora_B[t](self.lora_A[t](x))
        return out
    
class PolyMlp_OneLoRA(nn.Module):
    """ MLP as used in PolyNet  CP decomposition
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            use_act = False,
            bias=True,
            drop=0.,
            use_conv=False,
            n_degree=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_alpha=True,
            use_spatial=False,
            rank=10,
            n_tasks=10,
            alpha_lora=1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.rank = rank
        self.hidden_features = hidden_features or in_features
        self.use_alpha = use_alpha
        self.use_act = use_act
        self.use_spatial = use_spatial
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.norm1 =norm_layer(self.hidden_features)
        self.norm3 =norm_layer(self.hidden_features)

        self.n_degree = n_degree
        self.hidden_features = hidden_features
        self.alpha_lora = alpha_lora

        self.U1 = linear_layer(self.in_features, self.hidden_features, bias=bias)
        self.lora_A_U1 = nn.ModuleList([linear_layer(self.in_features, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_U1 = nn.ModuleList([linear_layer(self.rank, self.hidden_features, bias=False) for _ in range(n_tasks)])
        self.U1_lora = Channel_Projection_OneLoRA(self.in_features, self.hidden_features, bias=bias, use_conv=use_conv, weight_initialization=True, rank=self.rank, n_tasks=n_tasks, alpha_lora=alpha_lora)

        self.U2 = linear_layer(self.in_features, self.hidden_features//8, bias=bias)
        self.lora_A_U2 = nn.ModuleList([linear_layer(self.in_features, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_U2 = nn.ModuleList([linear_layer(self.rank, self.hidden_features//8, bias=False) for _ in range(n_tasks)])
        self.U2_lora = Channel_Projection_OneLoRA(self.in_features, self.hidden_features//8, bias=bias, use_conv=use_conv, weight_initialization=True, rank=rank, n_tasks=n_tasks, alpha_lora=alpha_lora)



        self.U3 = linear_layer(self.hidden_features//8, self.hidden_features, bias=bias)
        self.lora_A_U3 = nn.ModuleList([linear_layer(self.hidden_features//8, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_U3 = nn.ModuleList([linear_layer(self.rank, self.hidden_features, bias=False) for _ in range(n_tasks)])
        self.U3_lora = Channel_Projection_OneLoRA(self.hidden_features//8, self.hidden_features, bias=bias, use_conv=use_conv, weight_initialization=True, rank=rank, n_tasks=n_tasks, alpha_lora=alpha_lora)

        self.C = linear_layer(self.hidden_features, self.out_features, bias=True) 
        self.lora_A_C = nn.ModuleList([linear_layer(self.hidden_features, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B_C = nn.ModuleList([linear_layer(self.rank, self.out_features, bias=False) for _ in range(n_tasks)])
        self.C_lora = Channel_Projection_OneLoRA(self.hidden_features, self.out_features, bias=True, use_conv=use_conv, weight_initialization=False, rank=rank, n_tasks=n_tasks, alpha_lora=alpha_lora) 
        self.drop2 = nn.Dropout(drop_probs[0])
        
        if self.use_act:
            self.act = act_layer()
        if self.use_spatial:
            self.spatial_shift = Spatial_Shift()
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        # self.init_weights()
    
    def init_weights(self):
        self.U1_lora.init_weights_old(self.U1)
        self.U2_lora.init_weights_old(self.U2)
        self.U3_lora.init_weights_old(self.U3)
        self.C_lora.init_weights_old(self.C)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        if self.use_spatial:               
            out1 = self.U1_lora(x, task, register_hook, get_feat, get_cur_feat)
            out2 = self.U2_lora(x, task, register_hook, get_feat, get_cur_feat)
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out3 = self.U3_lora(out2, task, register_hook, get_feat, get_cur_feat)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        else:
            out1 = self.U1_lora(x, task, register_hook, get_feat, get_cur_feat)         
            out2 = self.U2_lora(x, task, register_hook, get_feat, get_cur_feat)
            out3 = self.U3_lora(out2, task, register_hook, get_feat, get_cur_feat)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
        else:
            out1 = out1 + out_so
        if self.use_act:
            out1 = self.act(out1)
        out = self.C_lora(out1, task, register_hook, get_feat, get_cur_feat)
        return out
    
class Channel_Projection_OneLoRA(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            use_conv=False,
            weight_initialization=False,
            rank=10,
            n_tasks=10,
            alpha_lora=1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.rank = rank
        self.alpha_lora = alpha_lora
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.matrix = torch.zeros(self.in_features ,self.in_features)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(self.in_features ,self.in_features)
        self.n_cur_matrix = 0
        self.U = None 
        self.lora_A = nn.ModuleList([linear_layer(self.in_features, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B = nn.ModuleList([linear_layer(self.rank, self.out_features, bias=False) for _ in range(n_tasks)])
    
    def init_weights_old(self, U_old): # change function name to prevent being called at hasattr(module, 'init_weights') in monet
        self.U = U_old

    def init_param(self):
        for t in range(len(self.lora_A)):
            nn.init.kaiming_uniform_(self.lora_A[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[t].weight)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        batch_size, h, w, channels = x.shape
        # GPM
        x_lora = x.reshape(batch_size, h * w, channels)
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_matrix += x_lora.shape[0]*x_lora.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_cur_matrix += x_lora.shape[0]*x_lora.shape[1]

        out = self.U(x)
        if task > -0.5:
            weight_lora = torch.stack([torch.mm(self.lora_B[t].weight, self.lora_A[t].weight) for t in range(task, task+1)], dim=0).sum(dim=0)
            x_lora = F.linear(x, weight_lora)
        out += self.alpha_lora * x_lora
        # for t in range(task+1):
        #     out += self.alpha_lora * self.lora_B[t](self.lora_A[t](x))
        return out

class PolyMlp_LoRA(nn.Module):
    """ MLP as used in PolyNet  CP decomposition
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            use_act = False,
            bias=True,
            drop=0.,
            use_conv=False,
            n_degree=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_alpha=True,
            use_spatial=False,
            rank=10,
            n_tasks=10,
            alpha_lora=1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.rank = rank
        self.hidden_features = hidden_features or in_features
        self.use_alpha = use_alpha
        self.use_act = use_act
        self.use_spatial = use_spatial
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.norm1 =norm_layer(self.hidden_features)
        self.norm3 =norm_layer(self.hidden_features)
        self.bias = bias
        self.use_conv = use_conv
        self.n_tasks = n_tasks

        self.n_degree = n_degree
        self.hidden_features = hidden_features
        self.alpha_lora = alpha_lora

        self.U1 = linear_layer(self.in_features, self.hidden_features, bias=bias)

        self.U2 = linear_layer(self.in_features, self.hidden_features//8, bias=bias)



        self.U3 = linear_layer(self.hidden_features//8, self.hidden_features, bias=bias)

        self.C = linear_layer(self.hidden_features, self.out_features, bias=True) 
        self.drop2 = nn.Dropout(drop_probs[0])
        
        if self.use_act:
            self.act = act_layer()
        if self.use_spatial:
            self.spatial_shift = Spatial_Shift()
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
        # self.init_weights()
    
    def init_weights(self):
        pass

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        if self.use_spatial:               
            out1 = self.U1(x)
            out2 = self.U2(x)
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        else:
            out1 = self.U1(x)         
            out2 = self.U2(x)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
        else:
            out1 = out1 + out_so
        if self.use_act:
            out1 = self.act(out1)
        out = self.C(out1)
        return out

class PolyMlp_LoRA_BCD(PolyMlp_LoRA):
    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        self.U2_lora = Channel_Projection_LoRA_Old(self.in_features, self.hidden_features//8, bias=self.bias, use_conv=self.use_conv, weight_initialization=True, rank=self.rank, n_tasks=self.n_tasks, alpha_lora=self.alpha_lora)

        self.U3_lora = Channel_Projection_LoRA_Old(self.hidden_features//8, self.hidden_features, bias=self.bias, use_conv=self.use_conv, weight_initialization=True, rank=self.rank, n_tasks=self.n_tasks, alpha_lora=self.alpha_lora)

        self.C_lora = Channel_Projection_LoRA_Old(self.hidden_features, self.out_features, bias=True, use_conv=self.use_conv, weight_initialization=False, rank=self.rank, n_tasks=self.n_tasks, alpha_lora=self.alpha_lora) 

    def init_weights(self):
        self.U2_lora.init_weights_old(self.U2)
        self.U3_lora.init_weights_old(self.U3)
        self.C_lora.init_weights_old(self.C)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        if self.use_spatial:               
            out1 = self.U1(x)
            out2 = self.U2_lora(x, task, register_hook, get_feat, get_cur_feat)
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out3 = self.U3_lora(out2, task, register_hook, get_feat, get_cur_feat)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        else:
            out1 = self.U1(x)         
            out2 = self.U2_lora(x, task, register_hook, get_feat, get_cur_feat)
            out3 = self.U3_lora(out2, task, register_hook, get_feat, get_cur_feat)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
        else:
            out1 = out1 + out_so
        if self.use_act:
            out1 = self.act(out1)
        out = self.C_lora(out1, task, register_hook, get_feat, get_cur_feat)
        return out

class PolyMlp_LoRA_AC(PolyMlp_LoRA):
    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        self.U1_lora = Channel_Projection_LoRA_Old(self.in_features, self.hidden_features, bias=self.bias, use_conv=self.use_conv, weight_initialization=True, rank=self.rank, n_tasks=self.n_tasks, alpha_lora=self.alpha_lora)

        self.C_lora = Channel_Projection_LoRA_Old(self.hidden_features, self.out_features, bias=True, use_conv=self.use_conv, weight_initialization=False, rank=self.rank, n_tasks=self.n_tasks, alpha_lora=self.alpha_lora) 

    def init_weights(self):
        self.U1_lora.init_weights_old(self.U1)
        self.C_lora.init_weights_old(self.C)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        if self.use_spatial:               
            out1 = self.U1_lora(x, task, register_hook, get_feat, get_cur_feat)
            out2 = self.U2(x)
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        else:
            out1 = self.U1_lora(x, task, register_hook, get_feat, get_cur_feat)         
            out2 = self.U2(x)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
        else:
            out1 = out1 + out_so
        if self.use_act:
            out1 = self.act(out1)
        out = self.C_lora(out1, task, register_hook, get_feat, get_cur_feat)
        return out
    
class PolyMlp_LoRA_ABD(PolyMlp_LoRA):
    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        self.U1_lora = Channel_Projection_LoRA_Old(self.in_features, self.hidden_features, bias=self.bias, use_conv=self.use_conv, weight_initialization=True, rank=self.rank, n_tasks=self.n_tasks, alpha_lora=self.alpha_lora)

        self.U2_lora = Channel_Projection_LoRA_Old(self.in_features, self.hidden_features//8, bias=self.bias, use_conv=self.use_conv, weight_initialization=True, rank=self.rank, n_tasks=self.n_tasks, alpha_lora=self.alpha_lora)

        self.U3_lora = Channel_Projection_LoRA_Old(self.hidden_features//8, self.hidden_features, bias=self.bias, use_conv=self.use_conv, weight_initialization=True, rank=self.rank, n_tasks=self.n_tasks, alpha_lora=self.alpha_lora)

    def init_weights(self):
        self.U1_lora.init_weights_old(self.U1)
        self.U2_lora.init_weights_old(self.U2)
        self.U3_lora.init_weights_old(self.U3)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        if self.use_spatial:               
            out1 = self.U1_lora(x, task, register_hook, get_feat, get_cur_feat)
            out2 = self.U2_lora(x, task, register_hook, get_feat, get_cur_feat)
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out3 = self.U3_lora(out2, task, register_hook, get_feat, get_cur_feat)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        else:
            out1 = self.U1_lora(x, task, register_hook, get_feat, get_cur_feat)       
            out2 = self.U2_lora(x, task, register_hook, get_feat, get_cur_feat)
            out3 = self.U3_lora(out2, task, register_hook, get_feat, get_cur_feat)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
        else:
            out1 = out1 + out_so
        if self.use_act:
            out1 = self.act(out1)
        out = self.C(out1)
        return out
    
class PolyMlp_LoRA_C(PolyMlp_LoRA):
    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        self.C_lora = Channel_Projection_LoRA_Old(self.hidden_features, self.out_features, bias=True, use_conv=self.use_conv, weight_initialization=False, rank=self.rank, n_tasks=self.n_tasks, alpha_lora=self.alpha_lora) 

    def init_weights(self):
        self.C_lora.init_weights_old(self.C)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        if self.use_spatial:               
            out1 = self.U1(x)
            out2 = self.U2(x)
            out1 = self.spatial_shift(out1)
            out2 = self.spatial_shift(out2)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        else:
            out1 = self.U1(x)         
            out2 = self.U2(x)
            out3 = self.U3(out2)
            out1 = self.norm1(out1)
            out3 = self.norm3(out3)
            out_so = out1 * out3
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
        else:
            out1 = out1 + out_so
        if self.use_act:
            out1 = self.act(out1)
        out = self.C_lora(out1, task, register_hook, get_feat, get_cur_feat)
        return out