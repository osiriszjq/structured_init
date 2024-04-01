""" ConvMixer

"""
import torch
import torch.nn as nn

import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d
from ._registry import register_model, generate_default_cfgs
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint_seq

__all__ = ['ConvMixer']

# initialization of convolusion kernel, C*1*K*K
def SpatialConv2d_init(C, kernel_size, init='random'):
    weight = None
    if (init == 'random')|(init == 'softmax'):
        weight = 1/kernel_size*(2*torch.rand((C,1,kernel_size,kernel_size))-1)
    elif init == 'impulse':
        k = torch.randint(0,kernel_size*kernel_size,(C,1))
        weight = torch.zeros((C,1,kernel_size*kernel_size))
        for i in range(C):
            for j in range(1):
                weight[i,j,k[i,j]] = 1
        weight = np.sqrt(1/kernel_size)*weight.reshape(C,1,kernel_size,kernel_size)
    elif init[:3] == 'box':
        weight = torch.zeros((C,1,kernel_size*kernel_size))
        for i in range(C):
            for j in range(1):
                k = np.random.choice(kernel_size*kernel_size,int(init[3:]),replace=False)
                weight[i,j,k] = 1
        weight = np.sqrt(1/int(init[3:])/kernel_size)*weight.reshape(C,1,kernel_size,kernel_size)
    elif init[:3] == 'gau':
        k = torch.randint(0,kernel_size,(C,1,2))
        weight = torch.zeros((C,1,kernel_size,kernel_size))
        for i in range(C):
            for j in range(1):
                for p in range(kernel_size):
                    for q in range(kernel_size):
                        weight[i,j,p,q] = (-0.5/float(init[3:])*((p-k[i,j,0])**2+(q-k[i,j,1])**2)).exp()
        weight = weight/((weight.flatten(1,3)**2).sum(1).mean()*kernel_size).sqrt()
    if weight is None:
        return -1
    else:
        return weight


# my spatial conv fuction, group=#channels, heads controls the number of different conv filters
class SpatialConv2d(nn.Module):
    def __init__(self, C, kernel_size, bias=True, init='random', num_heads = -1, trainable= True, input_weight=None):
        super(SpatialConv2d, self).__init__()
        self.C = C
        self.kernel_size = kernel_size
        self.init = init
        
        # different initialisation
        weight = SpatialConv2d_init(C,kernel_size,init=init)
        
        # how many heads or different filters we want to use
        if (num_heads<1)|(num_heads>C) :
            num_heads = C
        self.choice_idx = np.random.choice(num_heads,C,replace=(num_heads<C))

        # if use gloabal weight
        if input_weight is None:
            self.weight = nn.Parameter(weight[:num_heads],requires_grad=trainable)
        else:
            self.weight = input_weight

        if bias:
            bias = 1/kernel_size*(2*torch.rand((C))-1)
            self.bias = nn.Parameter(bias,requires_grad=trainable)
        else:
            self.bias = None


    def forward(self, x):
        if self.init == 'softmax':
            w_s = self.weight.shape
            return torch.nn.functional.conv2d(x, self.weight.flatten(2,3).softmax(-1).reshape(w_s)[self.choice_idx],self.bias,padding='same',groups=self.C)
        else:
            return torch.nn.functional.conv2d(x, self.weight[self.choice_idx], self.bias,padding='same',groups=self.C)




class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            kernel_size=9,
            patch_size=7,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            drop_rate=0.,
            act_layer=nn.GELU,
            init='random',
            num_heads=0,
            trainable=True,
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        self.grad_checkpointing = False

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            act_layer(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        # nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        SpatialConv2d(dim,kernel_size,init=init,num_heads=num_heads,trainable=trainable),
                        act_layer(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    act_layer(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )
        self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^stem', blocks=r'^blocks\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
          
    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.pooling(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_convmixer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for ConvMixer models.')

    return build_model_with_cfg(ConvMixer, variant, pretrained, **kwargs)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        'first_conv': 'stem.0',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'convmixer_1536_20.in1k': _cfg(hf_hub_id='timm/'),
    'convmixer_768_32.in1k': _cfg(hf_hub_id='timm/'),
    'convmixer_1024_20_ks9_p14.in1k': _cfg(hf_hub_id='timm/')
})



@register_model
def convmixer_1536_20(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=1536, depth=20, kernel_size=9, patch_size=7, **kwargs)
    return _create_convmixer('convmixer_1536_20', pretrained, **model_args)


@register_model
def convmixer_768_32(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, act_layer=nn.ReLU, **kwargs)
    return _create_convmixer('convmixer_768_32', pretrained, **model_args)


@register_model
def convmixer_1024_20_ks9_p14(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=1024, depth=20, kernel_size=9, patch_size=14, **kwargs)
    return _create_convmixer('convmixer_1024_20_ks9_p14', pretrained, **model_args)



# my small one for cifar

@register_model
def convmixer(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(**kwargs)
    return _create_convmixer('convmixer', pretrained, **model_args)


@register_model
def convmixer_256_8_3_train(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=256, depth=8, kernel_size=3, patch_size=2, **kwargs)
    return _create_convmixer('convmixer_256_8_3_train', pretrained, **model_args)
@register_model
def convmixer_256_8_3_random(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=256, depth=8, kernel_size=3, patch_size=2, trainable=False, **kwargs)
    return _create_convmixer('convmixer_256_8_3_random', pretrained, **model_args)
@register_model
def convmixer_256_8_3_impulse(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=256, depth=8, kernel_size=3, patch_size=2, init='box1', trainable=False, **kwargs)
    return _create_convmixer('convmixer_256_8_3_impulse', pretrained, **model_args)
@register_model
def convmixer_256_8_3_box(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=256, depth=8, kernel_size=3, patch_size=2, init='box9', trainable=False, **kwargs)
    return _create_convmixer('convmixer_256_8_3_box', pretrained, **model_args)




@register_model
def convmixer_512_6_3_train(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=512, depth=6, kernel_size=3, patch_size=2, **kwargs)
    return _create_convmixer('convmixer_512_6_3_train', pretrained, **model_args)
@register_model
def convmixer_512_6_3_random(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=512, depth=6, kernel_size=3, patch_size=2, trainable=False, **kwargs)
    return _create_convmixer('convmixer_512_6_3_random', pretrained, **model_args)
@register_model
def convmixer_512_6_3_impulse(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=512, depth=6, kernel_size=3, patch_size=2, init='box1', trainable=False, **kwargs)
    return _create_convmixer('convmixer_512_6_3_impulse', pretrained, **model_args)
@register_model
def convmixer_512_6_3_box(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=512, depth=6, kernel_size=3, patch_size=2, init='box9', trainable=False, **kwargs)
    return _create_convmixer('convmixer_512_6_3_box', pretrained, **model_args)




@register_model
def convmixer_512_6_5_train(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=512, depth=6, kernel_size=5, patch_size=2, **kwargs)
    return _create_convmixer('convmixer_512_6_5_train', pretrained, **model_args)
@register_model
def convmixer_512_6_5_random(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=512, depth=6, kernel_size=5, patch_size=2, trainable=False, **kwargs)
    return _create_convmixer('convmixer_512_6_5_random', pretrained, **model_args)
@register_model
def convmixer_512_6_5_impulse(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=512, depth=6, kernel_size=5, patch_size=2, init='box1', trainable=False, **kwargs)
    return _create_convmixer('convmixer_512_6_5_impulse', pretrained, **model_args)
@register_model
def convmixer_512_6_5_box(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=512, depth=6, kernel_size=5, patch_size=2, init='box25', trainable=False, **kwargs)
    return _create_convmixer('convmixer_512_6_5_box', pretrained, **model_args)