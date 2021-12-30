# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ActionNet.spatial_transforms import *
from ActionNet.temporal_transforms import *
from ActionNet import models as TSN_model

from .modeling_resnet import ResNetV2


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout=None):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)]
        print('dropout used:{}'.format(dropout))
        self.width = width
        self.layers = layers

        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 dropout=None, joint=False, emb_dropout=0., hybrid=True):
        '''
        hybrid : 混合了ResNet的ViT
        '''
        super().__init__()
        self.hybrid = hybrid
        in_channels = 3
        n_patches = (input_resolution // patch_size) * (input_resolution // patch_size)
        if self.hybrid:
            self.hybrid_model = ModifiedResNetV2(layers=(3, 4, 6, 3), output_dim=1024, heads=32
                                                 , input_resolution=224, width=64)
            in_channels = 64 * 16
            grid_size = 14
            patch_size = 1
            n_patches = (input_resolution // 16) * (input_resolution // 16)
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size,
                               bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(n_patches + 1, width))
        self.dropout = nn.Dropout(emb_dropout)
        self.ln_pre = LayerNorm(width)
        self.emb_dropout = emb_dropout
        self.joint = joint
        if joint:
            print('=====using joint space-time====')
            self.time_embedding = nn.Parameter(scale * torch.randn(T, width))
        if emb_dropout > 0:
            print('emb_dropout:{}'.format(emb_dropout))

        ## Attention Blocks
        self.transformer = Transformer(width, layers, heads, dropout=dropout)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        if self.hybrid:
            x = self.hybrid_model(x)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        if self.joint:
            B = x.shape[0] // self.T
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=self.T)
            x = x + self.time_embedding.to(x.dtype)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=self.T)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        weight_bn = weight_bn.half()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class Att(nn.Module):
    def __init__(self, channels):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)
        return x_out1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, use_nam=False, use_action=False):
        ##use_action: 是否添加action
        super().__init__()
        self.use_nam = use_nam
        self.use_action = use_action

        if self.use_action:
            self.in_channels = inplanes
            self.reduced_channels = self.in_channels // 16
            self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应池化结果为1x1
            self.sigmoid = nn.Sigmoid()

            # ste
            self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False,
                                             padding=(1, 1, 1))

            # ce
            self.action_p2_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1),
                                               stride=(1, 1),
                                               bias=False, padding=(0, 0))
            self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1,
                                             bias=False, padding=1,
                                             groups=1)
            self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1),
                                              stride=(1, 1),
                                              bias=False, padding=(0, 0))
            # me
            self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
            self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1),
                                               stride=(1, 1),
                                               bias=False, padding=(0, 0))
            self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
            self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3),
                                             stride=(1, 1), bias=False, padding=(1, 1), groups=self.reduced_channels)
            self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1),
                                              stride=(1, 1),
                                              bias=False, padding=(0, 0))

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if self.use_nam:
            self.nam = Att(planes * 4)

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        if self.use_action:
            self.n_segment = 8
            nt, c, h, w = x.size()
            n_batch = nt // 8
            x_p1 = x.view(n_batch, 8, c, h, w).transpose(2, 1).contiguous()
            x_p1 = x_p1.mean(1, keepdim=True)
            x_p1 = self.action_p1_conv1(x_p1)
            x_p1 = x_p1.transpose(2, 1).contiguous().view(nt, 1, h, w)
            x_p1 = self.sigmoid(x_p1)
            x_p1 = x * x_p1 + x
            # x = x * x_p1 + x

            x_p2 = self.avg_pool(x)
            x_p2 = self.action_p2_squeeze(x_p2)
            nt, c, h, w = x_p2.size()
            x_p2 = x_p2.view(n_batch, self.n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,
                                                                                                 1).contiguous()  # (n,c/16,t)
            x_p2 = self.action_p2_conv1(x_p2)
            x_p2 = self.relu(x_p2)  # (n,c/16,t)
            x_p2 = x_p2.transpose(2, 1).contiguous().view(-1, c, 1, 1)  # (nt,c/16,1,1)
            x_p2 = self.action_p2_expand(x_p2)  # (nt,c,1,1) 扩张通道
            x_p2 = self.sigmoid(x_p2)
            x_p2 = x * x_p2 + x

            x3 = self.action_p3_squeeze(x)
            x3 = self.action_p3_bn1(x3)
            nt, c, h, w = x3.size()
            x3_plus0, _ = x3.view(n_batch, self.n_segment, c, h, w).split([self.n_segment - 1, 1], dim=1)
            # (n,t-1,c/16,h,w)  (n,1,c/16,h,w)
            x3_plus1 = self.action_p3_conv1(x3)
            _, x3_plus1 = x3_plus1.view(n_batch, self.n_segment, c, h, w).split([1, self.n_segment - 1], dim=1)
            # (n,1,c/16,h,w)  (n,t-1,c/16,h,w)
            x_p3 = x3_plus1 - x3_plus0  # 这里对应conv(x(t+1)) - x(t) 的操作 (n,t-1,c/16,h,w)

            x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)  # (n,t,c/16,h,w) 填充第二维
            x_p3 = self.avg_pool(x_p3.view(nt, c, h, w))  # (nt,c/16,1,1)

            x_p3 = self.action_p3_expand(x_p3)  # (nt,c,1,1)
            x_p3 = self.sigmoid(x_p3)
            x_p3 = x * x_p3 + x

            x = x_p1 + x_p2 + x_p3

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)

        # print('conv3=', self.conv3)
        # print('device=', out.device)
        # print('size=',out.size())
        out = self.conv3(out)
        # print('done===')
        out = self.bn3(out)
        # out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.use_nam:
            out = self.nam(out)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNetV2(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, use_sis=False, use_nam=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.use_sis = use_sis
        self.use_nam = use_nam


        if use_sis:
            # se in se
            self.act_in_act_conv = nn.Conv2d(64, 2048, kernel_size=3, stride=8)
            self.act_in_act_bn = nn.BatchNorm2d(2048)
            self.act_in_act_relu = nn.ReLU(inplace=True)

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride, use_nam=self.use_nam)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, use_nam=self.use_nam))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('debug7887:',x.size())
        # x = self.layer4(x)

        # if self.use_sis:
        #     x = x + x_act_in

        # x = self.attnpool(x)

        return x


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, use_sis=False, use_nam=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.use_sis = use_sis
        self.use_nam = use_nam

        # action head: ste
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False, padding=(1, 1, 1))
        self.sigmoid = nn.Sigmoid()

        if use_sis:
            # se in se
            self.act_in_act_conv = nn.Conv2d(64, 2048, kernel_size=3, stride=8)
            self.act_in_act_bn = nn.BatchNorm2d(2048)
            self.act_in_act_relu = nn.ReLU(inplace=True)

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride, use_nam=self.use_nam)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, use_nam=self.use_nam))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)

        if self.use_sis:
            ###STE add to the head
            nt, c, h, w = x.size()
            n_batch = nt // 8
            x_p1 = x.view(n_batch, 8, c, h, w).transpose(2, 1).contiguous()
            x_p1 = x_p1.mean(1, keepdim=True)
            x_p1 = self.action_p1_conv1(x_p1)
            x_p1 = x_p1.transpose(2, 1).contiguous().view(nt, 1, h, w)
            x_p1 = self.sigmoid(x_p1)
            x = x * x_p1 + x

        x = stem(x)

        if self.use_sis:
            x_act_in = self.act_in_act_relu(self.act_in_act_bn(self.act_in_act_conv(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_sis:
            x = x + x_act_in

        x = self.attnpool(x)

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int, joint=False,
                 tsm=False, T=8, dropout=0., emb_dropout=0.,
                 modifiedRN=True,
                 use_sis=False,
                 use_nam=False
                 ):
        super().__init__()

        modifiedRN = True

        self.context_length = context_length
        if dropout > 0.:
            dpr = [x.item() for x in torch.linspace(0, dropout, vision_layers)]  # stochastic depth decay rule
        else:
            dpr = None

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64

            if modifiedRN:
                self.visual = ModifiedResNet(
                    layers=vision_layers,
                    output_dim=embed_dim,
                    heads=vision_heads,
                    input_resolution=image_resolution,
                    width=vision_width,
                    use_sis=use_sis,
                    use_nam=use_nam
                )


            else:
                ###ResNet 50
                self.visual = TSN_model.TSN(
                    101, 8, 'RGB',
                    is_shift=True,
                    partial_bn=True,
                    base_model='resnet50',
                    shift_div=8,
                    dropout=0.5,
                    img_feature_dim=224,
                    pretrain='imagenet',  # 'imagenet' or False
                    consensus_type='avg',
                    fc_lr5=True
                )
                self.visual.new_fc = nn.Linear(2048, 1024)
                # import torchvision.models as models
                # self.visual = models.resnet50(pretrained=True)
                # self.visual.fc = nn.Linear(2048, 1024)
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim, joint=joint, dropout=dpr,
                emb_dropout=emb_dropout
            )

        if tsm:
            print('=========using TSM==========')
            from modules.temporal_shift import make_temporal_shift_vit
            make_temporal_shift_vit(self.visual, T)

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            dropout=dpr
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.dropout = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        ##adapt for tsn
        # return self.visual.base_model.conv1.weight.dtype
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        ret = self.visual(image.type(self.dtype))
        # 128*512 （batch=16）d
        # print('encoded image size: ', ret.size())
        return ret

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.Conv3d, nn.GroupNorm)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, tsm=False, T=8, dropout=0., joint=False, emb_dropout=0., pretrain=True,
                is_action=False, use_sis=False, use_nam=False):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)

        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, tsm=tsm, T=T, joint=joint,
        dropout=dropout, emb_dropout=emb_dropout, modifiedRN=False, use_sis=use_sis, use_nam=use_nam
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    if tsm:
        for k in list(state_dict.keys()):
            if k.find("conv1") > -1 and k.find("layer") > -1:
                n_k = k.split('conv1.')[0] + 'conv1.net.' + k.split('conv1.')[1]
                state_dict[n_k] = state_dict.pop(k)
            if k.find("resblocks") > -1 and k.find("visual") > -1:
                tmp = ''
                for i, t_ in enumerate(k.split('resblocks.')[1].split('.')):
                    if i >= 1:
                        tmp += '.' + t_

                n_k = k.split('resblocks.')[0] + 'resblocks.' + k.split('resblocks.')[1].split('.')[0] + '.net' + tmp
                #                 print(n_k)
                state_dict[n_k] = state_dict.pop(k)

    convert_weights(model)
    if pretrain:
        print('loading clip pretrained model!')
        if joint:  # or emb_dropout>0 or dropout>0
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
    else:
        print('not using full clip pretrained model, only visual!')

        for k in list(state_dict.keys()):
            if not k.find("visual") > -1:
                state_dict.pop(k)
        del state_dict['visual.conv1.weight']
        from clip.clip import load_statedict
        h_state_dict = load_statedict('RN50')
        model.visual.hybrid_model.load_state_dict(h_state_dict, strict=False)
        model.load_state_dict(state_dict, strict=False)

    ##test
    # import torchvision.models as modles
    # model.visual = modles.resnet50(pretrained=True)
    ###为model添加Action-net模块，注意此时使用的是RN50，如果以后要求适应各种model，应当对此处进行修改。
    # if is_action:
    #     print('Adding action...')
    #     from ActionNet.action import make_temporal_shift
    #     # 用于添加ACTION模块
    #     make_temporal_shift(model.visual, T, n_div=8, place='blockres', temporal_pool=False)
    #     convert_weights(model)
    return model.eval()
