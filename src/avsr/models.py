"""Neural classifiers.

One architecture family is used for every stream so that comparisons isolate the *input*:
a temporal CNN treats each feature dimension as a channel and convolves along time with
dilated 1-D kernels, then pools over time (mean + max) so the position of the word inside the
3.8 s clip does not matter. Audio (100 Hz), visual (30 Hz) and early-fusion (100 Hz) inputs differ
only in channel count and length.

``SpectroTemporalCNN`` is the 2-D CNN used in the original audio-only system (MFCCs treated as
an image); it is kept as a baseline. ``ROICNN`` learns visual features end-to-end from pixels
for comparison with the hand-crafted DCT front-end.
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.drop(torch.relu(self.bn(self.conv(x)))) + self.skip(x)


class TemporalCNN(nn.Module):
    """``(B, D, T)`` -> logits ``(B, n_classes)``."""

    def __init__(self, in_dim: int, n_classes: int, channels: int = 128, kernel: int = 5,
                 dilations: tuple[int, ...] = (1, 2, 4, 8), dropout: float = 0.2, input_dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_dim)
        self.input_drop = nn.Dropout(input_dropout)
        layers, c = [], in_dim
        for d in dilations:
            layers.append(TemporalBlock(c, channels, kernel, d, dropout))
            c = channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * channels, channels), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(channels, n_classes))

    def embed(self, x):
        h = self.body(self.input_drop(self.input_norm(x)))
        return torch.cat([h.mean(-1), h.amax(-1)], dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


class SpectroTemporalCNN(nn.Module):
    """2-D CNN over the (feature x time) 'image' - the original coursework-style audio model."""

    def __init__(self, in_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.body = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, n_classes))

    def forward(self, x):                      # x: (B, D, T)
        h = self.body(x.unsqueeze(1))
        return self.head(h.mean(dim=(2, 3)))


class MLP(nn.Module):
    """For utterance-level (time-averaged) features - reproduces the original visual baseline."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(in_dim), nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):                      # x: (B, D, T) -> pooled over time
        return self.net(x.mean(-1))


class ROICNN(nn.Module):
    """End-to-end lip reader: per-frame 2-D CNN encoder followed by the temporal CNN."""

    def __init__(self, n_classes: int, roi_hw: tuple[int, int] = (32, 64), embed: int = 64):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.frontend = nn.Sequential(block(1, 16), block(16, 32), block(32, 64),
                                      nn.AdaptiveAvgPool2d((1, 2)), nn.Flatten(),
                                      nn.Linear(128, embed), nn.ReLU())
        self.temporal = TemporalCNN(embed, n_classes, channels=96, input_dropout=0.0)

    def forward(self, x):                      # x: (B, T, H, W) in [0, 1]
        B, T, H, W = x.shape
        z = self.frontend(x.reshape(B * T, 1, H, W)).reshape(B, T, -1)
        return self.temporal(z.transpose(1, 2))


def build_model(kind: str, in_dim: int, n_classes: int, **kw) -> nn.Module:
    if kind == "tcn":
        return TemporalCNN(in_dim, n_classes, **kw)
    if kind == "cnn2d":
        return SpectroTemporalCNN(in_dim, n_classes, **kw)
    if kind == "mlp":
        return MLP(in_dim, n_classes, **kw)
    if kind == "roicnn":
        return ROICNN(n_classes, **kw)
    raise ValueError(f"unknown model '{kind}'")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
