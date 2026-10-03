"""Backend contract and a tiny CPU flow model for executable infrastructure tests."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray  # [frames, channels], float32; never normalize/clamp here
    sample_rate: int


class FlowBackend(Protocol):
    decoder: nn.Module
    device: torch.device

    def prepare_reference(self, mode: str) -> None: ...
    def condition(self, record: dict) -> Any: ...
    def latent_shape(self, condition: Any) -> tuple[int, ...]: ...
    def velocity(self, x: torch.Tensor, t: float, condition: Any,
                 *, reference: bool = False) -> torch.Tensor: ...
    def decode(self, latent: torch.Tensor) -> DecodedAudio: ...
    def provenance(self) -> dict: ...


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 8)
        self.up_proj = nn.Linear(8, 8)
        self.down_proj = nn.Linear(8, 2)

    def forward(self, x, t, condition):
        h = torch.tanh(self.q_proj(x) + condition + float(t) * 0.1)
        return self.down_proj(torch.tanh(self.up_proj(h)))


class ToyBackend:
    """Uses no pretrained weights or music data; never an ACE-Step benchmark."""
    def __init__(self, *, device="cpu", duration_s=30.0):
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("The infrastructure-only toy backend intentionally runs on CPU")
        # Do not perturb the caller's global RNG while constructing the fixture.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            self.decoder = TinyDecoder().to(self.device)
        self.duration_s = duration_s
        self._reference = None

    def prepare_reference(self, mode):
        self._reference = copy.deepcopy(self.decoder).eval().requires_grad_(False)

    def condition(self, record):
        digest = hashlib.sha256(record["caption"].encode()).digest()
        return torch.tensor(int.from_bytes(digest[:2], "big") / 65535.0 - 0.5)

    def latent_shape(self, condition):
        return (1, 32, 2)

    def velocity(self, x, t, condition, *, reference=False):
        if reference and self._reference is None:
            raise RuntimeError("Reference must be frozen before configuring the policy")
        model = self._reference if reference else self.decoder
        return model(x, t, condition)

    def decode(self, latent):
        audio = F.interpolate(latent.transpose(1, 2), size=int(self.duration_s * 1000),
                              mode="linear", align_corners=False)
        return DecodedAudio(audio[0].transpose(0, 1).detach().cpu().float().numpy(), 1000)

    def provenance(self):
        return {"backend": "synthetic_cpu_fixture", "music_quality_evidence": False,
                "frozen_reference": self._reference is not None}
