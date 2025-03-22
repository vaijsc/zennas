# NOTE timm.models.layers is DEPRECATED, please use timm.layers, this is here to reduce breakages in transition
from timm_monet.layers.activations import *
from timm_monet.layers.adaptive_avgmax_pool import \
    adaptive_avgmax_pool2d, select_adaptive_pool2d, AdaptiveAvgMaxPool2d, SelectAdaptivePool2d
from timm_monet.layers.attention_pool2d import AttentionPool2d, RotAttentionPool2d, RotaryEmbedding
from timm_monet.layers.blur_pool import BlurPool2d
from timm_monet.layers.classifier import ClassifierHead, create_classifier
from timm_monet.layers.cond_conv2d import CondConv2d, get_condconv_initializer
from timm_monet.layers.config import is_exportable, is_scriptable, is_no_jit, set_exportable, set_scriptable, set_no_jit,\
    set_layer_config
from timm_monet.layers.conv2d_same import Conv2dSame, conv2d_same
from timm_monet.layers.conv_bn_act import ConvNormAct, ConvNormActAa, ConvBnAct
from timm_monet.layers.create_act import create_act_layer, get_act_layer, get_act_fn
from timm_monet.layers.create_attn import get_attn, create_attn
from timm_monet.layers.create_conv2d import create_conv2d
from timm_monet.layers.create_norm import get_norm_layer, create_norm_layer
from timm_monet.layers.create_norm_act import get_norm_act_layer, create_norm_act_layer, get_norm_act_layer
from timm_monet.layers.drop import DropBlock2d, DropPath, drop_block_2d, drop_path
from timm_monet.layers.eca import EcaModule, CecaModule, EfficientChannelAttn, CircularEfficientChannelAttn
from timm_monet.layers.evo_norm import EvoNorm2dB0, EvoNorm2dB1, EvoNorm2dB2,\
    EvoNorm2dS0, EvoNorm2dS0a, EvoNorm2dS1, EvoNorm2dS1a, EvoNorm2dS2, EvoNorm2dS2a
from timm_monet.layers.fast_norm import is_fast_norm, set_fast_norm, fast_group_norm, fast_layer_norm
from timm_monet.layers.filter_response_norm import FilterResponseNormTlu2d, FilterResponseNormAct2d
from timm_monet.layers.gather_excite import GatherExcite
from timm_monet.layers.global_context import GlobalContext
from timm_monet.layers.helpers import to_ntuple, to_2tuple, to_3tuple, to_4tuple, make_divisible, extend_tuple
from timm_monet.layers.inplace_abn import InplaceAbn
from timm_monet.layers.linear import Linear
from timm_monet.layers.mixed_conv2d import MixedConv2d
from timm_monet.layers.mlp import Mlp, GluMlp, GatedMlp, ConvMlp
from timm_monet.layers.non_local_attn import NonLocalAttn, BatNonLocalAttn
from timm_monet.layers.norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d
from timm_monet.layers.norm_act import BatchNormAct2d, GroupNormAct, convert_sync_batchnorm
from timm_monet.layers.padding import get_padding, get_same_padding, pad_same
from timm_monet.layers.patch_embed import PatchEmbed
from timm_monet.layers.pool2d_same import AvgPool2dSame, create_pool2d
from timm_monet.layers.squeeze_excite import SEModule, SqueezeExcite, EffectiveSEModule, EffectiveSqueezeExcite
from timm_monet.layers.selective_kernel import SelectiveKernel
from timm_monet.layers.separable_conv import SeparableConv2d, SeparableConvNormAct
from timm_monet.layers.space_to_depth import SpaceToDepthModule
from timm_monet.layers.split_attn import SplitAttn
from timm_monet.layers.split_batchnorm import SplitBatchNorm2d, convert_splitbn_model
from timm_monet.layers.std_conv import StdConv2d, StdConv2dSame, ScaledStdConv2d, ScaledStdConv2dSame
from timm_monet.layers.test_time_pool import TestTimePoolHead, apply_test_time_pool
from timm_monet.layers.trace_utils import _assert, _float_to_int
from timm_monet.layers.weight_init import trunc_normal_, trunc_normal_tf_, variance_scaling_, lecun_normal_

import warnings
warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", DeprecationWarning)
