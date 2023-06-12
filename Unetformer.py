import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Any
import xformers
import xformers.ops
from einops import rearrange
from ResNet import ResNet, BasicBlock


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled()
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [
            x.detach().requires_grad_(True) for x in ctx.input_tensors
        ]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class Upsample(nn.Module):
    """
    channel不减少，只增加尺寸
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels,
                                  self.out_channels,
                                  kernel_size=3,
                                  padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    channel不翻倍，只减少尺寸
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(self.channels,
                                self.out_channels,
                                3,
                                stride=stride,
                                padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_checkpoint=True,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels,
                      self.out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
        )
        self.updown = up or down
        # 绝对上采样还是下采样
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            # 都不是就给个空
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels), nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels,
                      self.out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1))

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels,
                                             self.out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels,
                                             self.out_channels,
                                             kernel_size=1,
                                             stride=1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        h = self.out_layers(h)
        return self.skip_connection(x) + h


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self,
                 query_dim,
                 condition_dim,
                 heads=4,
                 dim_head=64,
                 dropout=0.0):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, condition_dim is {condition_dim} and using "
            f"{heads} heads.")
        inner_dim = dim_head * heads

        self.heads = heads
        self.dim_head = dim_head
        self.condition_dim = condition_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(condition_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(condition_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                    nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, condition):
        if condition is None:
            condition = x
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c", h=h, w=h).contiguous()
        condition = rearrange(condition, "b c h w -> b (h w) c").contiguous()
        q = self.to_q(x)
        k = self.to_k(condition)
        v = self.to_v(condition)

        b, _, _ = q.shape
        # t.shape[1] -> (h w)
        # unsqueeze -> [b , (h w) , c] -> [b , (h w) , c , 1]
        # .reshape(b, t.shape[1], self.heads, self.dim_head) -> split heads ->  [b , (h w) , num_heads ,  head_dim]
        # .permute(0, 2, 1, 3) -> [b ,num_heads, (h w) , head_dim]
        # .reshape(b * self.heads, t.shape[1], self.dim_head) -> [b*num_heads, (h w) , head_dim]
        q, k, v = map(
            lambda t: t.unsqueeze(3).reshape(b, t.shape[
                1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(
                    b * self.heads, t.shape[1], self.dim_head).contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q,
                                                      k,
                                                      v,
                                                      attn_bias=None,
                                                      op=self.attention_op)

        out = (out.unsqueeze(0).reshape(
            b, self.heads, out.shape[1],
            self.dim_head).permute(0, 2, 1,
                                   3).reshape(b, out.shape[1],
                                              self.heads * self.dim_head))
        # self.to_out -> [b ,(h w) c]
        return self.to_out(out), h, w


class AttentionBlock(nn.Module):
    def __init__(self,
                 query_dim,
                 condition_dim,
                 height,
                 width,
                 heads=4,
                 dim_head=64,
                 dropout=0.0):
        super().__init__()
        self.attention = MemoryEfficientCrossAttention(query_dim,
                                                       condition_dim,
                                                       heads=heads,
                                                       dim_head=dim_head,
                                                       dropout=dropout)
        self.conv1 = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm([query_dim, height, width])

    def forward(self, x, condition):
        x_ = x
        x_, h, w = self.attention(x, condition)
        x_ = rearrange(x_, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x_ = x_ + x
        x_ = self.norm(x_)
        return self.conv1(x_) + x_


"""
attention = MemoryEfficientCrossAttention(64, 128).to("cuda")
x = torch.randn((1, 64, 100, 100)).to("cuda").float()
cc = torch.randn((1, 128, 200, 200)).to("cuda").float()
print(attention(x, cc).shape)
"""


class Unetformer(nn.Module):
    def __init__(self,
                 image_size,
                 input_channels,
                 model_channels,
                 num_classes,
                 dropout=.0,
                 resnetBlock=None):
        super().__init__()
        h, w = image_size
        if resnetBlock is None:
            resnetBlock = [2, 2, 2, 2]
        self.resnet = ResNet(BasicBlock, resnetBlock, include_top=False)
        self.input_block = nn.Conv2d(input_channels,
                                     model_channels,
                                     kernel_size=1,
                                     stride=1)
        self.down_sample1 = ResBlock(model_channels,
                                     dropout=dropout,
                                     down=True,
                                     out_channels=model_channels * 2)
        self.down_sample2 = ResBlock(model_channels * 2,
                                     dropout=dropout,
                                     down=True,
                                     out_channels=model_channels * 4)
        self.condition_down = Downsample(model_channels, use_conv=True)
        self.condition_down2 = Downsample(model_channels * 2, use_conv=True)
        stage1_layers = []
        for _ in range(3):
            stage1_layers.append(ResBlock(model_channels, dropout=dropout))
        stage1_layers.append(
            AttentionBlock(model_channels, 512, height=h, width=w))
        self.stage1 = nn.ModuleList(stage1_layers)

        stage2_layers = []
        for _ in range(3):
            stage2_layers.append(ResBlock(model_channels * 2, dropout=dropout))
        stage2_layers.append(
            AttentionBlock(model_channels * 2,
                           model_channels,
                           height=h // 2,
                           width=w // 2))
        self.stage2 = nn.ModuleList(stage2_layers)

        stage3_layers = []
        for _ in range(3):
            stage3_layers.append(ResBlock(model_channels * 4, dropout=dropout))
        stage3_layers.append(
            AttentionBlock(model_channels * 4,
                           model_channels * 2,
                           height=h // 4,
                           width=w // 4))
        self.stage3 = nn.ModuleList(stage3_layers)

        self.up_sample1 = ResBlock(model_channels * 4,
                                   dropout=dropout,
                                   up=True,
                                   out_channels=model_channels * 2)

        self.up_sample2 = ResBlock(model_channels * 2 + model_channels * 2,
                                   dropout=dropout,
                                   up=True,
                                   out_channels=model_channels * 2)

        self.res = ResBlock(model_channels * 2 + model_channels,
                            dropout=dropout,
                            out_channels=model_channels)

        self.attention = AttentionBlock(model_channels, model_channels, h, w)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc1 = nn.Linear(model_channels, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        condition = self.resnet(x)
        h1 = self.input_block(x)
        h2 = self.down_sample1(h1)
        h3 = self.down_sample2(h2)

        for module in self.stage1:
            if isinstance(module, AttentionBlock):
                h1 = module(h1, condition)
            else:
                h1 = module(h1)

        condition1 = self.condition_down(h1)
        for module in self.stage2:
            if isinstance(module, AttentionBlock):
                h2 = module(h2, condition1)
            else:
                h2 = module(h2)

        condition2 = self.condition_down2(h2)

        for module in self.stage3:
            if isinstance(module, AttentionBlock):
                h3 = module(h3, condition2)
            else:
                h3 = module(h3)

        h = self.up_sample1(h3)
        h = torch.concatenate([h, h2], dim=1)
        h = self.up_sample2(h)
        h = torch.concatenate([h, h1], dim=1)
        h = self.res(h)
        h = self.attention(h, condition=None)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        h = self.fc1(h)
        h = self.fc2(h)
        return h
