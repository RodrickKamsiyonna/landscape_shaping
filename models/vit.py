# adapted from:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
from einops import rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def generate_mask_matrix(npatch, nwindow):
    """Block-causal mask: frame i can attend to frames 0..i.

    Returns a bool tensor of shape (1, 1, npatch * nwindow, npatch * nwindow),
    where True means "allowed to attend".
    """
    zeros = torch.zeros(npatch, npatch, dtype=torch.bool)
    ones = torch.ones(npatch, npatch, dtype=torch.bool)
    rows = []
    for i in range(nwindow):
        rows.append(torch.cat(
            [ones] * (i + 1) + [zeros] * (nwindow - i - 1),
            dim=1,
        ))
    return torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,
                 num_patches=1, num_frames=1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out else nn.Identity()
        )
        self.register_buffer(
            "bias",
            generate_mask_matrix(num_patches, num_frames),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, k, v),
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Force the mask to bool regardless of what dtype the buffer ended up
        # with (float32 / bf16 / int after a resume, .to(dtype), etc.).
        # `~` is only defined for bool/integer tensors.
        allowed = self.bias[:, :, :T, :T].to(device=dots.device, dtype=torch.bool)
        dots = dots.masked_fill(~allowed, float("-inf"))

        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout=0., num_patches=1, num_frames=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    num_patches=num_patches,
                    num_frames=num_frames,
                ),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, depth, heads,
                 mlp_dim, pool='cls', dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, \
            "pool type must be either cls (cls token) or mean (mean pooling)"

        self.num_patches = num_patches
        self.num_frames = num_frames
        self.pool = pool

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            num_patches=num_patches,
            num_frames=num_frames,
        )

    def forward(self, x):
        _, n, _ = x.shape
        if n > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Input sequence length {n} exceeds maximum "
                f"{self.pos_embedding.shape[1]}"
            )

        x = x + self.pos_embedding[:, :n]
        return self.transformer(self.dropout(x))
