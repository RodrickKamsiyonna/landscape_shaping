# adapted from:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
from einops import rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def generate_mask_matrix(npatch, nwindow):
    """
    Create a causal temporal attention mask.

    Layout:
        [frame_0 patches]
        [frame_1 patches]
        ...
        [frame_{nwindow-1} patches]

    A frame may attend to itself and all previous frames,
    but not future frames.

    Output shape:
        (1, 1, nwindow * npatch, nwindow * npatch)

    Example with nwindow=3:

        frame 0 -> frame 0
        frame 1 -> frame 0, frame 1
        frame 2 -> frame 0, frame 1, frame 2
    """
    if npatch <= 0:
        raise ValueError(
            f"npatch must be > 0, got {npatch}"
        )

    if nwindow <= 0:
        raise ValueError(
            f"nwindow must be > 0, got {nwindow}"
        )

    zeros = torch.zeros(
        npatch,
        npatch,
        dtype=torch.bool,
    )

    ones = torch.ones(
        npatch,
        npatch,
        dtype=torch.bool,
    )

    rows = []

    for i in range(nwindow):
        row = torch.cat(
            [ones] * (i + 1)
            + [zeros] * (nwindow - i - 1),
            dim=1,
        )
        rows.append(row)

    mask = torch.cat(
        rows,
        dim=0,
    )

    return mask.unsqueeze(0).unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.0,
    ):
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
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        num_patches=1,
        num_frames=1,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (
            heads == 1
            and dim_head == dim
        )

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(
            dim,
            inner_dim * 3,
            bias=False,
        )

        self.to_out = (
            nn.Sequential(
                nn.Linear(
                    inner_dim,
                    dim,
                ),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

        # ----------------------------------------------------------
        # IMPORTANT:
        # Register the mask as a buffer.
        #
        # This means:
        #   model.to("cuda:0") -> bias moves to cuda:0
        #   model.to("cuda:1") -> bias moves to cuda:1
        #
        # Accelerate/DDP will therefore place the mask correctly
        # for each process.
        # ----------------------------------------------------------
        mask = generate_mask_matrix(
            num_patches,
            num_frames,
        )

        self.register_buffer(
            "bias",
            mask,
            persistent=False,
        )

        self.num_patches = num_patches
        self.num_frames = num_frames

    def forward(self, x):
        """
        x shape:
            (B, T, C)

        where:
            T <= num_frames * num_patches
        """
        B, T, C = x.size()

        if T > self.bias.shape[-1]:
            raise ValueError(
                "Input sequence is longer than the "
                "configured attention mask: "
                f"T={T}, "
                f"max_T={self.bias.shape[-1]}, "
                f"num_frames={self.num_frames}, "
                f"num_patches={self.num_patches}"
            )

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(
            3,
            dim=-1,
        )

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b n (h d) -> b h n d",
                h=self.heads,
            ),
            qkv,
        )

        dots = torch.matmul(
            q,
            k.transpose(-1, -2),
        ) * self.scale

        # ----------------------------------------------------------
        # Causal temporal attention.
        #
        # self.bias is now a registered buffer, so it lives on the
        # same device as dots after model/device placement.
        # ----------------------------------------------------------
        mask = self.bias[
            :, :, :T, :T
        ]

        dots = dots.masked_fill(
            ~mask,
            float("-inf"),
        )

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(
            attn,
            v,
        )

        out = rearrange(
            out,
            "b h n d -> b n (h d)",
        )

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        num_patches=1,
        num_frames=1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            num_patches=num_patches,
                            num_frames=num_frames,
                        ),
                        FeedForward(
                            dim,
                            mlp_dim,
                            dropout=dropout,
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()

        assert pool in {
            "cls",
            "mean",
        }, (
            "pool type must be either "
            "cls (cls token) or "
            "mean (mean pooling)"
        )

        self.num_patches = num_patches
        self.num_frames = num_frames
        self.pool = pool

        # ----------------------------------------------------------
        # Positional embedding
        # ----------------------------------------------------------
        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                num_frames * num_patches,
                dim,
            )
        )

        self.dropout = nn.Dropout(
            emb_dropout
        )

        # ----------------------------------------------------------
        # Transformer
        #
        # Pass num_patches and num_frames explicitly instead of using
        # process-global variables.
        # ----------------------------------------------------------
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
        """
        x:
            (B, num_frames * num_patches, dim)

        Returns:
            (B, num_frames * num_patches, dim)
        """
        b, n, _ = x.shape

        if n > self.pos_embedding.shape[1]:
            raise ValueError(
                "Input sequence is longer than the "
                "configured positional embedding: "
                f"n={n}, "
                f"max_n={self.pos_embedding.shape[1]}, "
                f"num_frames={self.num_frames}, "
                f"num_patches={self.num_patches}"
            )

        x = (
            x
            + self.pos_embedding[:, :n]
        )

        x = self.dropout(x)

        x = self.transformer(x)

        return x
