import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from einops import rearrange

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DPTLayer(nn.Module):
    def __init__(self, d_model=256, heads=8, dropout=0.1):
        super().__init__()
        self.col_att = nn.MultiheadAttention(d_model, heads, dropout=dropout)
        self.row_att = nn.MultiheadAttention(d_model, heads, dropout=dropout)
        self.scale_att = nn.MultiheadAttention(d_model, heads, dropout=dropout)

        self.norm1 = nn.GroupNorm(32, d_model)
        self.norm2 = nn.GroupNorm(32, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffd = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(d_model, d_model, 3, padding=1)
        )
        self.norm3 = nn.GroupNorm(32, d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, srcs, pos_embeds, num_grids):
        """
        srcs: list of feature maps, each with shape [B, C, H, W]
        pos_embeds: list of positional embeddings, each with shape [B, C, H, W]
        num_grids: list of number of grids, each with shape [B, 1]

        return: list of feature maps, each with shape [B, C, H, W]
        """
        # column & row
        scale_feats = []
        output = []
        for level, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            src = src + pos_embed
            src2 = rearrange(src, 'b c h w -> h (b w) c')
            src2, _ = self.col_att(src2, src2, src2)
            src2 = rearrange(src2, 'h (b w) c -> b c h w', b = bs, w = w)
            src = src + src2
            src2 = rearrange(src, 'b c h w -> w (b h) c')
            src2, _  = self.row_att(src2, src2, src2)
            src2 = rearrange(src2, 'w (b h) c -> b c h w', b = bs, h = h)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            if level!= 0:
                src = F.interpolate(src, size=num_grids[0], mode='bilinear')
            scale_feats.append(src)

        # scale
        srcs = torch.stack(scale_feats)
        s, bs, c, h, w = srcs.shape
        srcs2 = rearrange(srcs, 's b c h w -> s (b h w) c')
        srcs2, _  = self.scale_att(srcs2, srcs2, srcs2)
        srcs2 = rearrange(srcs2, 's (b h w) c -> s b c h w', b = bs, h = h, w = w)
        srcs = srcs + self.dropout2(srcs2)

        # ffd
        for level, src in enumerate(srcs):
            src = self.norm2(src)
            if level!= 0:
                src = F.interpolate(src, size=num_grids[level], mode='bilinear')
            src = self.dropout3(self.ffd(src)) + src
            src = self.norm3(src)
            output.append(src)

        return output

class DPT(nn.Module):
    def __init__(self, tri_layer, depth=12):
        super().__init__()
        self.layers = _get_clones(tri_layer, depth)
        self._reset_parameters()
    
    def forward(self, src, pos_embeds, num_grids):
        for layer in self.layers:
            src = layer(src, pos_embeds, num_grids)
        return src
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)