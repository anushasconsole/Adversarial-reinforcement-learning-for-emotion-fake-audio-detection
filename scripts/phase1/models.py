#!/usr/bin/env python3
"""Model definitions for Phase 1 defenders and attackers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchaudio
from transformers import WavLMModel


class ConvBinaryClassifier(nn.Module):
    def __init__(self, channels: tuple[int, int, int] = (32, 64, 128), dropout: float = 0.2) -> None:
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                ]
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        x = waveforms.unsqueeze(1)
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class LFCCBinaryClassifier(nn.Module):
    def __init__(self, sample_rate: int = 16000, n_lfcc: int = 60, dropout: float = 0.2) -> None:
        super().__init__()
        self.lfcc = torch.nn.Sequential(
            torchaudio.transforms.LFCC(sample_rate=sample_rate, n_lfcc=n_lfcc),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        feats = self.lfcc(waveforms)
        x = feats.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.head(x).squeeze(-1)


class WavLMBinaryClassifier(nn.Module):
    def __init__(self, model_name: str = "microsoft/wavlm-base", freeze_backbone: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(model_name)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.config.hidden_size, 1),
        )
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        attention_mask = None
        if lengths is not None:
            max_len = waveforms.shape[-1]
            time = torch.arange(max_len, device=waveforms.device).unsqueeze(0)
            attention_mask = (time < lengths.unsqueeze(1)).long()

        outputs = self.backbone(input_values=waveforms, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SqueezeformerStyleAttacker(nn.Module):
    def __init__(self, epsilon: float, hidden_channels: int = 64) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2, groups=4),
            nn.ReLU(),
            SqueezeExcitation(hidden_channels),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        delta = self.net(waveforms.unsqueeze(1)).squeeze(1)
        return torch.clamp(delta * self.epsilon, -self.epsilon, self.epsilon)


class MambaStyleAttacker(nn.Module):
    def __init__(self, epsilon: float, hidden_size: int = 96) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.in_proj = nn.Linear(1, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        x = waveforms.unsqueeze(-1)
        x = self.in_proj(x)
        x, _ = self.gru(x)
        delta = self.out_proj(x).squeeze(-1)
        return torch.clamp(delta * self.epsilon, -self.epsilon, self.epsilon)


@dataclass
class DefenderBundle:
    name: str
    model: nn.Module


def build_defender_model(name: str, wavlm: dict | None = None) -> nn.Module:
    key = name.lower()
    if key == "wavlm":
        wavlm_cfg = wavlm or {}
        return WavLMBinaryClassifier(
            model_name=wavlm_cfg.get("model_name", "microsoft/wavlm-base"),
            freeze_backbone=bool(wavlm_cfg.get("freeze_backbone", True)),
            dropout=float(wavlm_cfg.get("dropout", 0.2)),
        )
    if key == "aasist":
        return ConvBinaryClassifier(channels=(32, 64, 128), dropout=0.2)
    if key == "rawnet2":
        return ConvBinaryClassifier(channels=(64, 128, 256), dropout=0.3)
    if key == "lcnn":
        return LFCCBinaryClassifier(sample_rate=16000, n_lfcc=60, dropout=0.2)
    raise ValueError(f"Unsupported defender model: {name}")


def build_attacker_model(name: str, epsilon: float) -> nn.Module:
    key = name.lower()
    if key == "squeezeformer":
        return SqueezeformerStyleAttacker(epsilon=epsilon)
    if key == "mamba":
        return MambaStyleAttacker(epsilon=epsilon)
    raise ValueError(f"Unsupported attacker model: {name}")
