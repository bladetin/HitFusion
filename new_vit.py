from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from einops.layers.torch import Rearrange
from torchsummary import summary

vit_emb_size: int = 256  
vit_num_heads: int = 4 
vit_patch_size: int = 16  
vit_in_channels: int = 1  
vit_depth: int = 6 
vit_forward_expansion = 4
vit_forward_drop_p = 0.
vit_drop_p = 0.


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 16, emb_size: int = 256):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size,
                      kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 256, num_heads: int = 4, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        qkv = rearrange(
            self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 256,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size=vit_emb_size, num_heads=vit_num_heads, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(emb_size=vit_emb_size, drop_p=vit_drop_p, forward_expansion=vit_forward_expansion, forward_drop_p=vit_forward_drop_p, **kwargs)
                           for _ in range(depth)])


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 16,
                 emb_size: int = 256,
                 depth: int = 6,
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels=vit_in_channels,
                           patch_size=vit_patch_size, emb_size=vit_emb_size),
            TransformerEncoder(depth, **kwargs),
        )

class NaPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 16, emb_size: int = 256):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size,
                      kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, dict):
        q = self.projection(dict["q"])
        v = self.projection(dict["v"])
        k = v
        return {"q": q, "k": k, "v": v}

class NaFirstResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, dict, **kwargs):
        res = dict["v"]
        x = self.fn(dict, **kwargs)
        x["v"] += res

        x["k"] = x["v"]
        return x

class NaSecondResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, dict, **kwargs):
        res = dict["v"]
        x = self.fn(dict, **kwargs)
        x["v"] += res

        x["k"] = x["v"]
        return x

class NaLayerNorm(nn.Module):
    def __init__(self, emb_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, dict):
        q = dict["q"]
        v = self.norm(dict["v"])
        k = v

        return {"q": q, "k": k, "v": v}

class NaMultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 256, num_heads: int = 4, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.q_linear = nn.Linear(emb_size, emb_size)
        self.k_linear = nn.Linear(emb_size, emb_size)
        self.v_linear = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, dict, mask=None) :

        q = rearrange(self.q_linear(dict["q"]), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.k_linear(dict["k"]), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.v_linear(dict["v"]), "b n (h d) -> b h n d", h=self.num_heads)

        queries, keys, values = q, k, v

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        
        k = out
        v = out

        return {"q": dict["q"], "k": k, "v": v}

class NaDropout(nn.Module):
    def __init__(self, p: float = 0.):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, dict):
        q = dict["q"]
        v = self.dropout(dict["v"])
        k = v
        return {"q": q, "k": k, "v": v}

class NaFeedForwardBlock(nn.Module):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
    def forward(self, dict):

        q = dict["q"]
        v = self.layer(dict["v"])
        k = v

        return {"q": q, "k": k, "v": v}

class NaTransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 256,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            NaFirstResidualAdd(nn.Sequential(
                NaLayerNorm(emb_size),
                NaMultiHeadAttention(emb_size=vit_emb_size, num_heads=vit_num_heads, **kwargs),
                NaDropout(drop_p),
            )),
            NaSecondResidualAdd(nn.Sequential(
                NaLayerNorm(emb_size),
                NaFeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                NaDropout(drop_p)
            )
            ))

class NaTransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(*[NaTransformerEncoderBlock(emb_size=vit_emb_size, drop_p=vit_drop_p, forward_expansion=vit_forward_expansion, forward_drop_p=vit_forward_drop_p, **kwargs)
                           for _ in range(depth)])

class NaViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 16,
                 emb_size: int = 256,
                 depth: int = 6,
                 **kwargs):
        super().__init__(
            NaPatchEmbedding(in_channels=vit_in_channels,
                           patch_size=vit_patch_size, emb_size=vit_emb_size),
            NaTransformerEncoder(depth, **kwargs),
        )
    

if __name__ == "__main__":

    x1_ir = torch.randn(32, 1, 640, 480)
    x1_vis = torch.randn(32, 1, 640, 480)
    dict1 = {'q': x1_ir, 'k': x1_vis, 'v': x1_vis}

    x2_ir = torch.randn(32, 1, 224, 224)
    x2_vis = torch.randn(32, 1, 224, 224)
    dict2 = {'q': x2_ir, 'k': x2_vis, 'v': x2_vis}

    navit = NaViT()
    print(navit)

    y1 = navit(dict1)
    y2 = navit(dict2)

    print(y1["v"].shape)
    print(y2["v"].shape)

