import torch
from torch import nn
from einops import rearrange

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, input):
        for attend, feedforward in self.layers:
            input = attend(input) + input
            input = feedforward(input) + input
        return input


class PreNorm(nn.Module):
    def __init__(self, dim, module):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module

    def forward(self, input):
        return self.module(self.norm(input))


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Indentity()

    def forward(self, input):
        qkv = self.to_qkv(input).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange (t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention = self.attend(dots)
        output = torch.matmul(attention, v)
        output = rearrange(output, 'b p h n d -> b p n (h d)')
        return self.to_out(output)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return self.net(input)