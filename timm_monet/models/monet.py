import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Reduce
from einops import rearrange

from timm_monet.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm_monet.layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple
from timm_monet.layers.mlp import PolyMlp, PolyMlp_SkipLoRA, PolyMlp_SkipOneLoRA, PolyMlp_Weighted_SkipLoRA, PolyMlp_LoRA_Old, PolyMlp_OneLoRA, PolyMlp_LoRA_BCD, PolyMlp_LoRA_AC, PolyMlp_LoRA_ABD, PolyMlp_LoRA_C
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, checkpoint_seq
from ._registry import register_model

__all__ = ['PolyBlock'] 


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, 1, 1, dim), 1e-7))
        # self.beta= nn.Parameter(torch.full((1, 1, 1, dim), 1e-6))
        # self.alpha = nn.Parameter(torch.ones((1, 1, 1,dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, 1,dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)
    
class PolyBlock(nn.Module):
    def __init__(
            self,
            embed_dim,
            expansion_factor = 3,
            # mlp_layer = PolyMlp_NCPv2,
            mlp_layer = PolyMlp,
            # norm_layer=Affine,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer = None,
            drop=0.,
            drop_path=0.,
            n_degree = 2, # second order interaction
            use_act = False,
            n_tasks = 10,
            rank = 64,
            lora_in_dim = 3136,
            alpha_lora=1,
            **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.norm = norm_layer(self.embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp1 = mlp_layer(self.embed_dim, self.embed_dim, self.embed_dim, act_layer=act_layer,drop=drop,use_spatial=True,use_act=use_act,rank=rank,n_tasks=n_tasks, alpha_lora=alpha_lora)
        self.mlp2= mlp_layer(self.embed_dim, self.embed_dim*self.expansion_factor, self.embed_dim,act_layer=act_layer, drop=drop,use_spatial=False,use_act=use_act,rank=rank,n_tasks=n_tasks, alpha_lora=alpha_lora)
    
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        z = self.norm(x)
        z = self.mlp1(z, task, register_hook, get_feat, get_cur_feat)  
        x = x + self.drop_path(z)
        z = self.norm(x)
        z = self.mlp2(z, task, register_hook, get_feat, get_cur_feat)
        x = x + self.drop_path(z)
        return x
    
class PolyBlock_LoRA(nn.Module):
    def __init__(
            self,
            embed_dim,
            expansion_factor = 3,
            # mlp_layer = PolyMlp_NCPv2,
            mlp_layer = PolyMlp,
            # norm_layer=Affine,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer = None,
            drop=0.,
            drop_path=0.,
            n_degree = 2, # second order interaction
            use_act = False,
            n_tasks = 10,
            rank = 64,
            lora_in_dim = 3136,
            alpha_lora=1,
            **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.rank = rank
        self.norm = norm_layer(self.embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp1 = mlp_layer(self.embed_dim, self.embed_dim, self.embed_dim, act_layer=act_layer,drop=drop,use_spatial=True,use_act=use_act,rank=rank,n_tasks=n_tasks)
        self.mlp2= mlp_layer(self.embed_dim, self.embed_dim*self.expansion_factor, self.embed_dim,act_layer=act_layer, drop=drop,use_spatial=False,use_act=use_act,rank=rank,n_tasks=n_tasks)
        self.lora_A = nn.ModuleList([nn.Linear(lora_in_dim, self.rank, bias=False) for _ in range(n_tasks)])
        self.lora_B = nn.ModuleList([nn.Linear(self.rank, lora_in_dim, bias=False) for _ in range(n_tasks)])
        self.matrix = torch.zeros(lora_in_dim ,lora_in_dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(lora_in_dim ,lora_in_dim)
        self.n_cur_matrix = 0

    def init_param(self):
        for t in range(len(self.lora_A)):
            nn.init.kaiming_uniform_(self.lora_A[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[t].weight)
    
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        z = self.norm(x)

        batch_size, h, w, channels = z.shape
        z_lora = z.permute(0, 3, 1, 2)
        z_lora = z_lora.reshape(batch_size, channels, h * w)
        
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(z_lora.detach().permute(0, 2, 1), z_lora.detach()).sum(dim=0).cpu())/(self.n_matrix + z_lora.shape[0]*z_lora.shape[1])
            self.n_matrix += z_lora.shape[0]*z_lora.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(z_lora.detach().permute(0, 2, 1), z_lora.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + z_lora.shape[0]*z_lora.shape[1])
            self.n_cur_matrix += z_lora.shape[0]*z_lora.shape[1]

        if task > -0.5:
            weight_lora = torch.stack([torch.mm(self.lora_B[t].weight, self.lora_A[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            z_lora = F.linear(z_lora, weight_lora)

        z_lora = z_lora.reshape(batch_size, channels, h, w)
        z_lora = z_lora.permute(0, 2, 3, 1)

        z = self.mlp1(z, task, register_hook, get_feat, get_cur_feat)  
        x = x + self.drop_path(z)
        z = self.norm(x)
        z = self.mlp2(z, task, register_hook, get_feat, get_cur_feat)
        x = x + self.drop_path(z) + z_lora
        return x

# Custom Sequential class that handles additional arguments for PolyBlock
class SequentialWithArgs(nn.Sequential):
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        # Pass additional arguments to each module in the Sequential container
        for module in self:
            x = module(x, task, register_hook, get_feat, get_cur_feat)
        return x

class basic_blocks(nn.Module):
    def __init__(self,index,layers,embed_dim, expansion_factor = 4, dropout = 0., drop_path = 0.,norm_layer=partial(nn.LayerNorm, eps=1e-6),act_layer = nn.GELU,use_act = False, poly_block = PolyBlock, mlp_layer = PolyMlp, lora_in_dim = 3136, alpha_lora=1, **kwargs):
        super().__init__()
        self.poly_block = poly_block
        self.mlp_layer = mlp_layer
        # Using the custom Sequential class to wrap PolyBlock and handle arguments correctly
        self.model = SequentialWithArgs(
            *[SequentialWithArgs(
                self.poly_block(
                    embed_dim=embed_dim, 
                    expansion_factor=expansion_factor, 
                    drop=dropout, 
                    drop_path=drop_path,
                    use_act=use_act,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_layer = self.mlp_layer,
                    lora_in_dim = lora_in_dim,
                    alpha_lora = alpha_lora,
                    **kwargs
                )
            ) for _ in range(layers[index])]
        )
    
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.model(x, task, register_hook, get_feat, get_cur_feat)
        x = rearrange(x, 'b h w c -> b c h w')
        return x

class Downsample(nn.Module):
    """ Downsample transition stage   design for pyramid structure
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size, rank=10,n_tasks = 10, alpha_lora=1):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        # x = rearrange(x, 'b c h w -> b h w c')
        x = self.proj(x)
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = x.permute(0, 3, 1, 2)
        # x = self.proj(x)  # B, C, H, W
        # x = x.permute(0, 2, 3, 1)
        return x
    
class Downsample_LoRA(nn.Module):
    """ Downsample transition stage   design for pyramid structure
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size, rank=10, n_tasks = 10, alpha_lora=1):
        super().__init__()
        assert patch_size == 2, patch_size
        self.rank = rank
        self.alpha_lora = alpha_lora
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)
        # self.lora_A = nn.ModuleList([nn.Conv2d(in_embed_dim, self.rank, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False) for _ in range(n_tasks)])
        # self.lora_B = nn.ModuleList([nn.Conv2d(self.rank, out_embed_dim, kernel_size=1, stride=1, padding=0, bias=False) for _ in range(n_tasks)])
        self.lora_A = nn.ModuleList([nn.Linear(in_embed_dim, self.rank, bias=False) for _ in range(n_tasks)])
        # self.lora_A = nn.ModuleList([nn.Conv2d(in_embed_dim, self.rank, kernel_size=1, stride=1, padding=0, bias=False) for _ in range(n_tasks)])
        self.lora_B = nn.ModuleList([nn.Conv2d(self.rank, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False) for _ in range(n_tasks)])

        self.matrix = torch.zeros(in_embed_dim, in_embed_dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(in_embed_dim, in_embed_dim)
        self.n_cur_matrix = 0

    def init_param(self):
        for t in range(len(self.lora_A)):
            nn.init.kaiming_uniform_(self.lora_A[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[t].weight)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        x_bhwc = rearrange(x, 'b c h w -> b h w c')
        out = self.proj(x)

        x_lora = rearrange(x_bhwc, 'b h w c -> b (h w) c')
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_matrix += x_lora.shape[0]*x_lora.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_cur_matrix += x_lora.shape[0]*x_lora.shape[1]

        for t in range(task+1):
            temp_lora_A = self.lora_A[t](x_bhwc)
            temp_lora_A = rearrange(temp_lora_A, 'b h w c -> b c h w')
            out += self.alpha_lora * self.lora_B[t](temp_lora_A)
            
        return out
    
class Downsample_Weighted_LoRA(nn.Module):
    """ Downsample transition stage   design for pyramid structure
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size, rank=10, n_tasks = 10, alpha_lora=1):
        super().__init__()
        assert patch_size == 2, patch_size
        self.rank = rank
        self.alpha_lora = alpha_lora
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)
        # self.lora_A = nn.ModuleList([nn.Conv2d(in_embed_dim, self.rank, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False) for _ in range(n_tasks)])
        # self.lora_B = nn.ModuleList([nn.Conv2d(self.rank, out_embed_dim, kernel_size=1, stride=1, padding=0, bias=False) for _ in range(n_tasks)])
        self.lora_A = nn.ModuleList([nn.Linear(in_embed_dim, self.rank, bias=False) for _ in range(n_tasks)])
        # self.lora_A = nn.ModuleList([nn.Conv2d(in_embed_dim, self.rank, kernel_size=1, stride=1, padding=0, bias=False) for _ in range(n_tasks)])
        self.lora_B = nn.ModuleList([nn.Conv2d(self.rank, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False) for _ in range(n_tasks)])

        self.matrix = torch.zeros(in_embed_dim, in_embed_dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(in_embed_dim, in_embed_dim)
        self.n_cur_matrix = 0

    def init_param(self):
        for t in range(len(self.lora_A)):
            nn.init.kaiming_uniform_(self.lora_A[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[t].weight)

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        x_bhwc = rearrange(x, 'b c h w -> b h w c')
        out = self.proj(x)

        x_lora = rearrange(x_bhwc, 'b h w c -> b (h w) c')
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_matrix += x_lora.shape[0]*x_lora.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x_lora.detach().permute(0, 2, 1), x_lora.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x_lora.shape[0]*x_lora.shape[1])
            self.n_cur_matrix += x_lora.shape[0]*x_lora.shape[1]

        sum_lora = 0
        for t in range(task+1):
            temp_lora_A = self.lora_A[t](x_bhwc)
            temp_lora_A = rearrange(temp_lora_A, 'b h w c -> b c h w')
            sum_lora += self.lora_B[t](temp_lora_A)

        out = (1 - self.alpha_lora) * out + self.alpha_lora * sum_lora
            
        return out
    
class LayerNorm_LoRA(nn.Module):
    """LayerNorm with LoRA applied to the scale and bias."""
    def __init__(self, normalized_shape, eps=1e-6, rank=4):
        super(LayerNorm_LoRA, self).__init__()
        self.norm = partial(nn.LayerNorm, eps=eps)(normalized_shape)

        # LoRA for scaling (gain)
        self.lora_A_scale = nn.Linear(normalized_shape, rank, bias=False)
        self.lora_B_scale = nn.Linear(rank, normalized_shape, bias=False)

        # LoRA for bias
        self.lora_A_bias = nn.Linear(normalized_shape, rank, bias=False)
        self.lora_B_bias = nn.Linear(rank, normalized_shape, bias=False)

        # Alpha parameters to control LoRA adjustments
        self.alpha_scale = nn.Parameter(torch.ones(1))
        self.alpha_bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Apply standard LayerNorm
        normalized_output = self.norm(x)

        # LoRA adjustment for the scale (gain)
        lora_scale_adjustment = self.lora_B_scale(self.lora_A_scale(x)) * self.alpha_scale
        scale_adjusted_output = normalized_output * (1 + lora_scale_adjustment)

        # LoRA adjustment for the bias
        lora_bias_adjustment = self.lora_B_bias(self.lora_A_bias(x)) * self.alpha_bias
        final_output = scale_adjusted_output + lora_bias_adjustment

        return final_output

class MONet(nn.Module):
    def __init__(
        self,
        image_size=224,
        num_classes=1000,
        in_chans=3,
        patch_size= 2,
        mlp_ratio = [0.5, 4.0],
        block_layer =basic_blocks,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp,
        down_sample = Downsample,
        # norm_layer=Affine,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=None,
        drop_rate=0.,
        drop_path_rate=0.,
        nlhb=False,
        global_pool='avg',
        transitions = None,
        embed_dim=[192, 384],
        layers = None,
        expansion_factor = [3, 3],
        feature_fusion_layer = None,
        use_act = False,
        use_multi_level = False,
        lora_in_dims = None,
        rank = 64,
        n_tasks = 10,
        alpha_lora = 1
    ):
        # self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        # embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None,  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        # norm_layer=nn.LayerNorm, mlp_fn=CycleMLP, fork_feat=False
        self.num_classes = num_classes
        self.image_size = image_size
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim[-1]  # num_features for consistency with other models
        self.use_multi_level = use_multi_level
        self.grad_checkpointing = False
        self.layers = layers
        self.embed_dim = embed_dim
        self.poly_block = poly_block
        self.mlp_layer = mlp_layer
        self.rank = rank
        self.n_tasks = n_tasks
        self.alpha_lora = alpha_lora
        image_size = pair(self.image_size)
        oldps = [1, 1]
        for ps in patch_size:
            ps = pair(ps)
            oldps[0] = oldps[0] * ps[0]
            oldps[1] = oldps[1] * ps[1]
        super().__init__()
    
        self.fs = nn.Conv2d(in_chans, embed_dim[0], kernel_size=patch_size[0], stride=patch_size[0])
        self.fs2 = nn.Conv2d(embed_dim[0], embed_dim[0], kernel_size=2, stride=2)
        network = []
        assert len(layers) == len(embed_dim) == len(expansion_factor)
        for i in range(len(layers)):
            stage = block_layer(i,self.layers,embed_dim[i], expansion_factor[i], dropout = drop_rate,drop_path = drop_path_rate,norm_layer = norm_layer,act_layer = act_layer,use_act = use_act, poly_block = self.poly_block, mlp_layer = self.mlp_layer, lora_in_dim = lora_in_dims[i], rank = self.rank, n_tasks = self.n_tasks, alpha_lora = self.alpha_lora)
            network.append(stage)
            if i >= len(self.layers)-1:
                break
            if transitions[i] or embed_dim[i] != embed_dim[i+1]:
                patch_size = 2 if transitions[i] else 1
                network.append(down_sample(embed_dim[i], embed_dim[i+1], patch_size, rank = self.rank, n_tasks = self.n_tasks, alpha_lora = self.alpha_lora))
        self.network = nn.Sequential(*network)
        self.head = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(embed_dim[-1], self.num_classes)
        )
        self.init_weights(nlhb=nlhb)

    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False):
        x = self.fs(x)
        x = self.fs2(x)
        if self.use_multi_level:
            x2 = self.fs3(x)
            x = x + self.alpha1 * x2

        for i, blk in enumerate(self.network):
            x = blk(x, task_id, register_blk==i, get_feat=get_feat, get_cur_feat=get_cur_feat)
        
        return x
        

    @torch.jit.ignore
    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # num_blocks-first

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()



def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                torch.nn.init.kaiming_normal_(module.weight,a=0.001)
                # nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in num_blocks-first order.
        module.init_weights()



def _create_improved_MONet(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for MLP-Mixer models.')
    model = build_model_with_cfg(
        MONet, variant, pretrained,
        **kwargs)
    return model

#Multi-Stage MONet, design for ImageNet Resolution Image
@register_model
def MONet_T(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4, 8, 12, 10]  # real patch size [8,16,32,64]  [4,8,16,32]
    embed_dims = [64, 128, 192, 192]
    expansion_factor = [3, 3, 3, 3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        lora_in_dims = [3136, 784, 196, 49],
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_T', pretrained=pretrained, **model_args)
    return model


@register_model
def MONet_S(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        lora_in_dims = [3136, 784, 196, 49],
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_S', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_lora(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock_LoRA,
        mlp_layer = PolyMlp,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_lora', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_Projlora(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_LoRA_Old,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_Projlora', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_Projlora_DSlora(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_LoRA,
        down_sample = Downsample_LoRA,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_Projlora_DSlora', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_SkipLora(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_SkipLoRA,
        down_sample = Downsample_LoRA,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_SkipLora', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_NoDownLora_SkipLora(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_SkipLoRA,
        down_sample = Downsample,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_NoDownLora_SkipLora', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_NoDownLora_Weighted_SkipLora(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_Weighted_SkipLoRA,
        down_sample = Downsample,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_NoDownLora_SkipLora', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_DownLora_SkipLora(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_SkipLoRA,
        down_sample = Downsample_LoRA,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_NoDownLora_SkipLora', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_DownLora_Weighted_SkipLora(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_Weighted_SkipLoRA,
        down_sample = Downsample_Weighted_LoRA,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_NoDownLora_SkipLora', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_NoDownLora_SkipOneLora(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_SkipOneLoRA,
        down_sample = Downsample,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_NoDownLora_SkipOneLora', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_Proj_OneLora(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_OneLoRA,
        down_sample = Downsample,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_Proj_OneLora', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_Projlora_BCD(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_LoRA_BCD,
        down_sample = Downsample,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_Projlora_BCD', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_Projlora_AC(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_LoRA_AC,
        down_sample = Downsample,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_Projlora_AC', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_Projlora_ABD(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_LoRA_ABD,
        down_sample = Downsample,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_Projlora_ABD', pretrained=pretrained, **model_args)
    return model

@register_model
def MONet_Projlora_C(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [4,6,12,14]  # [4,8,16,32]
    embed_dims = [128,192,256,384]
    expansion_factor = [3, 3, 3, 3]
    
    dict_args = dict(
        patch_size=[2], 
        layers=layers,
        transitions=transitions,
        embed_dim=embed_dims,
        expansion_factor = expansion_factor,
        poly_block = PolyBlock,
        mlp_layer = PolyMlp_LoRA_C,
        down_sample = Downsample,
        lora_in_dims = [3136, 784, 196, 49], # width * height
        **kwargs
        )
    
    model_args = dict_args
    model = _create_improved_MONet('MONet_Projlora_C', pretrained=pretrained, **model_args)
    return model


