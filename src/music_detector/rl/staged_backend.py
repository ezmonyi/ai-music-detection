"""Inference-only component staging for a single low-VRAM CUDA collector.

No quantization, CPU inference, optimizer or sampler substitution. GPU fit and
cross-GPU likelihood replay still require a real hardware probe.
"""
from dataclasses import replace
import math
from collections.abc import Mapping

import torch

from .acestep_backend import (AceStepBackend, LATENT_CHANNELS, LATENT_HZ,
                             MIN_LATENT_FRAMES, SILENCE_REFERENCE_FRAMES, _last_hidden_state)


class StagedAceStepBackend(AceStepBackend):
    def __init__(self, cfg):
        target = torch.device(cfg.device)
        if target.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Staged collection requires an available explicit CUDA device")
        # Load on host first, avoiding an all-components-on-GPU startup peak.
        # _load_dtype below retains the original BF16 checkpoint dtype on host.
        super().__init__(replace(cfg, device="cpu"))
        self.cfg, self.device = cfg, target
        self.silence_latent = self.silence_latent.to(target)

    @property
    def _load_dtype(self):
        return torch.bfloat16 if self.cfg.precision == "bfloat16" else torch.float32

    def _stage(self, phase):
        if phase not in {"text", "condition", "flow", "decode"}:
            raise ValueError("Unknown collection phase")
        modules = {"model": self.model, "text": self.text_encoder, "vae": self.vae}
        active = {"text": {"text"}, "condition": {"model"},
                  "flow": {"model"}, "decode": {"vae"}}[phase]
        # Offload inactive modules before bringing another component on GPU.
        for name, module in modules.items():
            if name not in active:
                module.to("cpu")
        torch.cuda.empty_cache()
        for name in active:
            modules[name].to(self.device)  # Device only: preserve FP32 LoRA tensors.

    @torch.no_grad()
    def condition(self, record):
        # Same operations/token limits as the pinned adapter, but never keep
        # the 0.6B text encoder and 2B condition/flow model on GPU together.
        if not isinstance(record, Mapping) or not isinstance(record.get("caption"), str):
            raise ValueError("ACE-Step condition records require a string caption")
        duration = float(record.get("duration_s", 30.0))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_s must be finite and positive")
        frames = max(MIN_LATENT_FRAMES, int(round(duration * LATENT_HZ)))
        source_latents = self._tile_silence(frames)
        chunk_masks = torch.ones(1, frames, LATENT_CHANNELS, device=self.device, dtype=torch.bool)
        attention_mask = torch.ones(1, frames, device=self.device, dtype=torch.bool)
        self._stage("text")
        text = self.text_tokenizer(self._text_prompt(record), padding="longest", truncation=True,
                                   max_length=256, return_tensors="pt")
        lyric = self.text_tokenizer(self._lyric_prompt(record), padding="longest", truncation=True,
                                    max_length=2048, return_tensors="pt")
        text_mask = text.attention_mask.to(self.device).bool()
        lyric_mask = lyric.attention_mask.to(self.device).bool()
        text_hidden = _last_hidden_state(self.text_encoder(text.input_ids.to(self.device))).to(self._load_dtype)
        lyric_hidden = self.text_encoder.embed_tokens(lyric.input_ids.to(self.device)).to(self._load_dtype)
        self._stage("condition")
        encoder_hidden, encoder_mask, context = self.model.prepare_condition(
            text_hidden_states=text_hidden, text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hidden, lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=self._tile_silence(SILENCE_REFERENCE_FRAMES),
            refer_audio_order_mask=torch.zeros(1, device=self.device, dtype=torch.long),
            hidden_states=source_latents, attention_mask=attention_mask,
            silence_latent=self.silence_latent, src_latents=source_latents, chunk_masks=chunk_masks,
            is_covers=torch.zeros(1, device=self.device, dtype=torch.bool))
        result = {"encoder_hidden_states": encoder_hidden.detach(),
                  "encoder_attention_mask": encoder_mask.detach(), "context_latents": context.detach(),
                  "attention_mask": attention_mask.detach(), "src_latents": source_latents.detach(),
                  "null_condition_emb": self.model.null_condition_emb.detach().clone(), "latent_frames": frames}
        self._stage("flow")
        return result

    def velocity(self, x, t, condition, *, reference=False):
        if next(self.decoder.parameters()).device != self.device:
            self._stage("flow")
        return super().velocity(x, t, condition, reference=reference)

    def decode(self, latent):
        self._stage("decode")
        try:
            return super().decode(latent)
        finally:
            self.vae.to("cpu")
            torch.cuda.empty_cache()

    def prepare_reference(self, mode):
        if mode != "lora":
            raise ValueError("Low-VRAM collection supports the existing LoRA policy only")
        super().prepare_reference(mode)

    def provenance(self):
        return {**super().provenance(), "component_staging": "gpu_text_then_condition_then_flow_then_gpu_vae",
                "quantization": None, "cpu_forward": False,
                "gpu_fit_verified": False}
