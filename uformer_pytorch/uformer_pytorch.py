from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# constants

LayerNorm = partial(nn.InstanceNorm2d, affine = True)
List = nn.ModuleList

# helpers

def exists(val):
    return val is not None

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x, skip = None):
        h, y = self.heads, x.shape[-1]
        q = self.to_q(x)

        kv_input = x

        if exists(skip):
            kv_input = torch.cat((kv_input, skip), dim = 0)

        k, v = self.to_kv(kv_input).chunk(2, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        if exists(skip):
            k, v = map(lambda t: rearrange(t, '(r b) n d -> b (r n) d', r = 2), (k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = h, y = y)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        hidden_dim = dim * mult
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = List([])
        for _ in range(depth):
            self.layers.append(List([
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult))
            ]))

    def forward(self, x, skip = None):
        for attn, ff in self.layers:
            x = attn(x, skip = skip) + x
            x = ff(x) + x
        return x

# classes

class Uformer(nn.Module):
    def __init__(
        self,
        dim = 64,
        channels = 3,
        stages = 4,
        num_blocks = 2,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.project_in = nn.Sequential(
            nn.Conv2d(channels, dim, 3, padding = 1),
            nn.LeakyReLU()
        )

        self.project_out = nn.Sequential(
            nn.Conv2d(dim, channels, 3, padding = 1),
        )

        self.downs = List([])
        self.mid = Block(dim = dim * 2 ** stages, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult)
        self.ups = List([])

        for ind in range(stages):
            self.downs.append(List([
                Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult),
                nn.Conv2d(dim, dim * 2, 4, stride = 2, padding = 1)
            ]))

            self.ups.append(List([
                nn.ConvTranspose2d(dim * 2, dim, 2, stride = 2),
                Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult)
            ]))

            dim *= 2

    def forward(
        self,
        x
    ):
        x = self.project_in(x)

        skips = []
        for block, downsample in self.downs:
            x = block(x)
            skips.append(x)
            x = downsample(x)

        x = self.mid(x)

        for (upsample, block), skip in zip(reversed(self.ups), reversed(skips)):
            x = upsample(x)
            x = block(x, skip = skip)

        x = self.project_out(x)
        return x
