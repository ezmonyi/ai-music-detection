"""Local-only ACE-Step 1.5 backend for the online flow-policy trainer.

The adapter deliberately loads the reviewed upstream Python sources directly
from ``ModelConfig.upstream_dir`` and loads checkpoint artefacts only from the
explicit local checkpoint directory.  It never asks Hugging Face/Transformers
to resolve remote code or weights.  The text encoder, condition encoder and
VAE are frozen; only ``model.decoder`` is exposed to the policy configurator.

The sampler in :mod:`music_detector.rl.trainer` owns the Flow-GRPO transition
and log-probability calculation.  This module therefore exposes only a
condition cache, a differentiable decoder velocity call, and raw VAE decode.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import nn

from .backends import DecodedAudio
from .config import ModelConfig


AUDITED_UPSTREAM_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
AUDITED_SFT_CONFIG_COMMIT = "c410d249e71ea9385a7b586865e65b1473e1098d"
AUDITED_VAE_CONFIG_COMMIT = "19671f406d603126926c1b7e2adc169acbcade22"
AUDITED_FLOW_GRPO_COMMIT = "879042cf5707f8b90daa98d147d7deac2317c5da"
SAMPLE_RATE = 48_000
LATENT_HZ = 25
LATENT_CHANNELS = 64
VAE_DOWNSAMPLE = 1_920
MIN_LATENT_FRAMES = 128
SILENCE_REFERENCE_FRAMES = 750


def _path_is_inside(path: Path, parent: Path) -> bool:
    """Return whether *path* resolves below *parent*."""

    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    """Hash one local file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> dict[str, Any]:
    """Return a deterministic aggregate hash for a local artefact tree."""

    if not root.is_dir():
        raise FileNotFoundError(f"Local artefact directory does not exist: {root}")
    entries: list[tuple[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        entries.append((relative, _sha256_file(path)))
    digest = hashlib.sha256()
    for relative, file_hash in entries:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "file_count": len(entries)}


def _git_head(path: Path) -> str:
    """Read a local Git HEAD without contacting a remote."""

    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"upstream_dir is not a readable Git checkout: {path}")
    return result.stdout.strip()


def _assert_clean_pinned_checkout(path: Path, expected: str) -> None:
    """Require the exact reviewed commit and no local source modifications."""

    if not path.is_dir() or not (path / ".git").exists():
        raise ValueError(f"upstream_dir must be a local Git checkout: {path}")
    actual = _git_head(path)
    if actual != expected:
        raise ValueError(f"ACE-Step source commit mismatch: expected {expected}, got {actual}")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("upstream_dir must be a clean checkout of the audited source commit")


def _validate_sft_config(config: Mapping[str, Any]) -> None:
    """Validate the exact 2B-class non-Turbo SFT architecture contract."""

    expected = {
        "model_type": "acestep",
        "hidden_size": 2048,
        "in_channels": 192,
        "audio_acoustic_hidden_dim": 64,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 6144,
        "text_hidden_dim": 1024,
        "patch_size": 2,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported ACE-Step checkpoint architecture: {key}={config.get(key)!r}")
    if config.get("is_turbo") is not False:
        raise ValueError("ACE-Step RL adapter requires the non-Turbo SFT checkpoint")
    architectures = config.get("architectures")
    if architectures != ["AceStepConditionGenerationModel"]:
        raise ValueError(f"Unexpected ACE-Step architecture declaration: {architectures!r}")
    if config.get("num_audio_decoder_hidden_layers") != 24:
        raise ValueError("ACE-Step SFT decoder must have 24 hidden layers")


def _validate_vae_config(config: Mapping[str, Any]) -> None:
    """Validate the local 48 kHz stereo VAE contract before loading weights."""

    if config.get("sampling_rate") != SAMPLE_RATE:
        raise ValueError(f"ACE-Step VAE must be {SAMPLE_RATE} Hz")
    if config.get("audio_channels") != 2:
        raise ValueError("ACE-Step VAE must be stereo (audio_channels=2)")
    if config.get("decoder_input_channels") != LATENT_CHANNELS:
        raise ValueError("ACE-Step VAE decoder input must be 64 latent channels")
    ratios = config.get("downsampling_ratios")
    if not isinstance(ratios, list) or not ratios or any(type(value) is not int or value <= 0 for value in ratios):
        raise ValueError("ACE-Step VAE downsampling_ratios must be positive integers")
    product = 1
    for value in ratios:
        product *= value
    if product != VAE_DOWNSAMPLE:
        raise ValueError(f"ACE-Step VAE downsampling product must be {VAE_DOWNSAMPLE}, got {product}")
    if SAMPLE_RATE // product != LATENT_HZ:
        raise ValueError("ACE-Step VAE sample rate/downsampling do not produce the 25 Hz latent contract")


def _unpack_loading_info(result: Any, *, component: str) -> Any:
    """Unpack ``from_pretrained(output_loading_info=True)`` and fail closed.

    A missing or unexpected key can otherwise leave a decoder partially/randomly
    initialized while still producing tensors.  The policy trainer must never
    optimize against that silently degraded reference.
    """

    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[1], Mapping):
        raise RuntimeError(f"{component} loader did not return loading information")
    module, info = result
    problems: dict[str, Any] = {}
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = info.get(key)
        if values:
            problems[key] = values
    if problems:
        raise RuntimeError(f"Local {component} checkpoint has incompatible weights: {problems}")
    return module


def _last_hidden_state(output: Any) -> torch.Tensor:
    """Extract a Transformers-style last hidden state."""

    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError("text encoder output has no last_hidden_state tensor")


class AceStepBackend:
    """Frozen-condition, local-only ACE-Step 1.5 policy backend."""

    def __init__(self, cfg: ModelConfig):
        if not isinstance(cfg, ModelConfig):
            raise TypeError("AceStepBackend expects a ModelConfig")
        if cfg.backend != "acestep":
            raise ValueError("AceStepBackend requires ModelConfig.backend='acestep'")
        if not cfg.upstream_dir or not cfg.checkpoint_dir or not cfg.upstream_commit:
            raise ValueError("ACE-Step requires explicit upstream_dir, checkpoint_dir, and upstream_commit")
        if cfg.upstream_commit != AUDITED_UPSTREAM_COMMIT:
            raise ValueError(
                f"Only audited ACE-Step commit {AUDITED_UPSTREAM_COMMIT} is supported; "
                f"got {cfg.upstream_commit}"
            )

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested explicitly but is unavailable; refusing CPU fallback")
        self.upstream_dir = Path(cfg.upstream_dir).expanduser().resolve()
        self.checkpoint_dir = Path(cfg.checkpoint_dir).expanduser().resolve()
        self.model_dir = self.checkpoint_dir / cfg.checkpoint_name
        self.vae_dir = self.checkpoint_dir / "vae"
        self.text_dir = self.checkpoint_dir / "Qwen3-Embedding-0.6B"
        _assert_clean_pinned_checkout(self.upstream_dir, cfg.upstream_commit)
        for required in (self.model_dir, self.vae_dir, self.text_dir):
            if not required.is_dir():
                raise FileNotFoundError(f"Required local ACE-Step artefact is missing: {required}")

        config_path = self.model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"SFT config.json is missing: {config_path}")
        checkpoint_config = json.loads(config_path.read_text())
        _validate_sft_config(checkpoint_config)
        vae_config_path = self.vae_dir / "config.json"
        if not vae_config_path.is_file():
            raise FileNotFoundError(f"VAE config.json is missing: {vae_config_path}")
        vae_config = json.loads(vae_config_path.read_text())
        _validate_vae_config(vae_config)

        self._model_config_json = checkpoint_config
        self._vae_config_json = vae_config
        self._reference_mode: str | None = None
        self._reference_state: dict[str, torch.Tensor] | None = None
        self._reference_state_device: tuple[torch.device, dict[str, torch.Tensor]] | None = None
        self._load_runtime()

    @property
    def _load_dtype(self) -> torch.dtype:
        """Return a safe checkpoint/component dtype for the selected device."""

        if self.cfg.precision == "bfloat16" and self.device.type == "cuda":
            return torch.bfloat16
        return torch.float32

    @property
    def _compute_dtype(self) -> torch.dtype:
        """Return the requested CUDA autocast dtype, or fp32 elsewhere."""

        if self.cfg.precision == "bfloat16" and self.device.type == "cuda":
            return torch.bfloat16
        return torch.float32

    def _import_local_runtime(self):
        """Import reviewed model classes from the pinned checkout only."""

        package_dir = self.upstream_dir / "acestep"
        if not package_dir.is_dir():
            raise FileNotFoundError(f"Pinned ACE-Step source package is missing: {package_dir}")
        for module_name, existing in sys.modules.items():
            if not (module_name == "acestep" or module_name.startswith("acestep.")):
                continue
            module_file = getattr(existing, "__file__", None)
            if module_file and not _path_is_inside(Path(module_file), self.upstream_dir):
                raise RuntimeError("A different 'acestep' package is already imported; refusing source ambiguity")
        source = str(self.upstream_dir)
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            config_module = importlib.import_module("acestep.models.base.configuration_acestep_v15")
            model_module = importlib.import_module("acestep.models.base.modeling_acestep_v15_base")
        except Exception as exc:
            raise RuntimeError(
                "Could not import the pinned local ACE-Step implementation; install its declared "
                "runtime dependencies in the execution environment, without enabling remote code"
            ) from exc
        for module in (config_module, model_module):
            module_file = getattr(module, "__file__", None)
            if not module_file or not _path_is_inside(Path(module_file), self.upstream_dir):
                raise RuntimeError("ACE-Step runtime resolved outside the pinned upstream checkout")
        return config_module.AceStepConfig, model_module.AceStepConditionGenerationModel

    def _load_runtime(self) -> None:
        """Load the local model, VAE, text encoder, and silence latent."""

        config_cls, model_cls = self._import_local_runtime()
        try:
            from diffusers.models import AutoencoderOobleck
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "ACE-Step runtime dependencies are unavailable; no package or model download is attempted"
            ) from exc

        # Directly instantiate the reviewed local class.  This intentionally
        # does not call AutoModel.from_pretrained(..., trust_remote_code=True).
        model_config = config_cls.from_pretrained(str(self.model_dir), local_files_only=True)
        model = _unpack_loading_info(
            model_cls.from_pretrained(
                str(self.model_dir),
                config=model_config,
                local_files_only=True,
                output_loading_info=True,
                torch_dtype=self._load_dtype,
                attn_implementation="sdpa",
            ),
            component="SFT decoder",
        )
        model = model.to(device=self.device, dtype=self._load_dtype)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.decoder = model.decoder
        if hasattr(self.decoder, "config"):
            self.decoder.config.attention_dropout = 0.0
        self._enable_gradient_checkpointing()

        vae = _unpack_loading_info(
            AutoencoderOobleck.from_pretrained(
                str(self.vae_dir),
                local_files_only=True,
                output_loading_info=True,
                torch_dtype=self._load_dtype,
            ),
            component="VAE",
        )
        self.vae = vae.to(device=self.device, dtype=self._load_dtype).eval()
        self.vae.requires_grad_(False)

        self.text_tokenizer = AutoTokenizer.from_pretrained(str(self.text_dir), local_files_only=True)
        self.text_encoder = _unpack_loading_info(
            AutoModel.from_pretrained(
                str(self.text_dir),
                local_files_only=True,
                output_loading_info=True,
                torch_dtype=self._load_dtype,
                attn_implementation="sdpa",
            ),
            component="text encoder",
        )
        self.text_encoder = self.text_encoder.to(device=self.device, dtype=self._load_dtype).eval()
        self.text_encoder.requires_grad_(False)

        silence_path = self.model_dir / "silence_latent.pt"
        if not silence_path.is_file():
            raise FileNotFoundError(f"SFT silence_latent.pt is missing: {silence_path}")
        silence = torch.load(silence_path, map_location="cpu", weights_only=True)
        if not isinstance(silence, torch.Tensor) or silence.ndim != 3:
            raise ValueError("silence_latent.pt must contain a rank-3 tensor")
        self.silence_latent = silence.transpose(1, 2).contiguous().to(self.device, dtype=self._load_dtype)
        if self.silence_latent.shape[0] != 1 or self.silence_latent.shape[-1] != LATENT_CHANNELS:
            raise ValueError("silence_latent.pt must decode to [1,T,64]")

        self._artefact_hashes = {
            "model": _sha256_tree(self.model_dir),
            "vae": _sha256_tree(self.vae_dir),
            "text_encoder": _sha256_tree(self.text_dir),
        }

    def _enable_gradient_checkpointing(self) -> None:
        """Enable upstream checkpointing when requested and verify it took effect."""

        if not self.cfg.gradient_checkpointing:
            return
        enable = getattr(self.model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise RuntimeError("Pinned ACE-Step model does not expose gradient_checkpointing_enable")
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            enable()
        # The reviewed decoder checks this flag inside forward; Transformers'
        # generic helper may not set it on older compatible releases.
        if hasattr(self.decoder, "gradient_checkpointing"):
            self.decoder.gradient_checkpointing = True
        if not getattr(self.decoder, "gradient_checkpointing", False):
            raise RuntimeError("ACE-Step gradient checkpointing was requested but not enabled")

    def _tile_silence(self, frames: int) -> torch.Tensor:
        """Return silence latent with exactly ``frames`` temporal positions."""

        if frames <= 0:
            raise ValueError("latent frame count must be positive")
        source = self.silence_latent
        repeats = (frames + source.shape[1] - 1) // source.shape[1]
        return source[:, :source.shape[1], :].repeat(1, repeats, 1)[:, :frames, :]

    @staticmethod
    def _metadata_text(record: Mapping[str, Any]) -> str:
        """Format SFT metadata exactly like upstream ``MetadataMixin``."""

        metadata = record.get("metas", record.get("metadata"))
        if isinstance(metadata, str):
            # Explicit strings are already serialized SFT metadata; preserve
            # them instead of converting them to JSON or inventing fields.
            return metadata
        if not isinstance(metadata, Mapping):
            metadata = {}
        bpm = metadata.get("bpm", metadata.get("tempo", "N/A"))
        timesignature = metadata.get("timesignature", metadata.get("time_signature", "N/A"))
        keyscale = metadata.get("keyscale", metadata.get("key", metadata.get("scale", "N/A")))
        duration = metadata.get("duration", metadata.get("length", record.get("duration_s", 30)))
        if isinstance(duration, (int, float)):
            duration = f"{int(duration)} seconds"
        elif not isinstance(duration, str):
            duration = "30 seconds"
        return (
            f"- bpm: {bpm}\n"
            f"- timesignature: {timesignature}\n"
            f"- keyscale: {keyscale}\n"
            f"- duration: {duration}\n"
        )

    @staticmethod
    def _instruction(record: Mapping[str, Any]) -> str:
        instruction = str(record.get("instruction", "Fill the audio semantic mask based on the given conditions:"))
        return instruction if instruction.endswith(":") else instruction + ":"

    @classmethod
    def _text_prompt(cls, record: Mapping[str, Any]) -> str:
        return (
            "# Instruction\n"
            f"{cls._instruction(record)}\n\n"
            "# Caption\n"
            f"{record['caption']}\n\n"
            "# Metas\n"
            f"{cls._metadata_text(record)}<|endoftext|>\n"
        )

    @staticmethod
    def _lyric_prompt(record: Mapping[str, Any]) -> str:
        language = str(record.get("language", record.get("vocal_language", "unknown")))
        lyrics = str(record.get("lyrics", ""))
        return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

    @torch.no_grad()
    def condition(self, record: dict) -> dict[str, torch.Tensor | int | float]:
        """Build and cache upstream ACE-Step conditioning for one prompt."""

        if not isinstance(record, Mapping) or not isinstance(record.get("caption"), str):
            raise ValueError("ACE-Step condition records require a string caption")
        duration = float(record.get("duration_s", 30.0))
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration_s must be finite and positive")
        frames = max(MIN_LATENT_FRAMES, int(round(duration * LATENT_HZ)))
        source_latents = self._tile_silence(frames)
        chunk_masks = torch.ones(
            1, frames, LATENT_CHANNELS, device=self.device, dtype=torch.bool
        )
        attention_mask = torch.ones(1, frames, device=self.device, dtype=torch.bool)

        text_inputs = self.text_tokenizer(
            self._text_prompt(record),
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        lyric_inputs = self.text_tokenizer(
            self._lyric_prompt(record),
            padding="longest",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        text_ids = text_inputs.input_ids.to(self.device)
        lyric_ids = lyric_inputs.input_ids.to(self.device)
        text_mask = text_inputs.attention_mask.to(self.device).bool()
        lyric_mask = lyric_inputs.attention_mask.to(self.device).bool()
        # ``prepare_condition`` is itself upstream ``@torch.no_grad``.  Keep
        # this outer path in no-grad (rather than inference_mode): inference
        # tensors cannot later be saved by autograd when a trainable decoder
        # projection consumes the cached condition.
        with torch.no_grad():
            text_hidden = _last_hidden_state(self.text_encoder(text_ids)).to(self._load_dtype)
            lyric_hidden = self.text_encoder.embed_tokens(lyric_ids).to(self._load_dtype)
            # Text2music uses a silent reference for the frozen timbre branch.
            refer_latents = self._tile_silence(SILENCE_REFERENCE_FRAMES)
            refer_order = torch.zeros(1, device=self.device, dtype=torch.long)
            encoder_hidden, encoder_mask, context = self.model.prepare_condition(
                text_hidden_states=text_hidden,
                text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hidden,
                lyric_attention_mask=lyric_mask,
                refer_audio_acoustic_hidden_states_packed=refer_latents,
                refer_audio_order_mask=refer_order,
                hidden_states=source_latents,
                attention_mask=attention_mask,
                silence_latent=self.silence_latent,
                src_latents=source_latents,
                chunk_masks=chunk_masks,
                is_covers=torch.zeros(1, device=self.device, dtype=torch.bool),
            )
        return {
            "encoder_hidden_states": encoder_hidden.detach(),
            "encoder_attention_mask": encoder_mask.detach(),
            "context_latents": context.detach(),
            "attention_mask": attention_mask.detach(),
            "src_latents": source_latents.detach(),
            "null_condition_emb": self.model.null_condition_emb.detach(),
            "latent_frames": frames,
        }

    def latent_shape(self, condition: Mapping[str, Any]) -> tuple[int, int, int]:
        """Return the ACE latent shape consumed by the Flow-GRPO sampler."""

        context = condition.get("context_latents")
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError("condition must contain [B,T,128] context_latents")
        if context.shape[-1] != 2 * LATENT_CHANNELS:
            raise ValueError("ACE-Step context_latents must have 128 channels")
        return (int(context.shape[0]), int(context.shape[1]), LATENT_CHANNELS)

    def _expand_condition_tensor(self, value: torch.Tensor, batch: int, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Move a cached condition tensor and expand singleton batch dimensions."""

        result = value.to(self.device, dtype=dtype) if dtype is not None else value.to(self.device)
        if result.shape[0] == 1 and batch != 1:
            result = result.expand(batch, *result.shape[1:])
        if result.shape[0] != batch:
            raise ValueError(f"condition batch {result.shape[0]} does not match latent batch {batch}")
        return result

    @contextmanager
    def _autocast(self) -> Iterator[None]:
        """Use the configured CUDA compute dtype while preserving fp32 masters."""

        if self.device.type == "cuda" and self._compute_dtype != torch.float32:
            with torch.autocast(device_type="cuda", dtype=self._compute_dtype):
                yield
        else:
            with nullcontext():
                yield

    def _reference_state_for_device(self) -> dict[str, torch.Tensor]:
        """Materialize the immutable full-decoder reference state on device."""

        if self._reference_state is None:
            raise RuntimeError("Full reference state was not prepared")
        if self._reference_state_device is not None and self._reference_state_device[0] == self.device:
            return self._reference_state_device[1]
        current = dict(self.decoder.named_parameters())
        current.update(dict(self.decoder.named_buffers()))
        state: dict[str, torch.Tensor] = {}
        for name, value in self._reference_state.items():
            target = current.get(name)
            if target is None:
                raise RuntimeError(f"Reference state name is absent from current decoder: {name}")
            state[name] = value.to(device=self.device, dtype=target.dtype if value.is_floating_point() else value.dtype)
        self._reference_state_device = (self.device, state)
        return state

    def _decoder_call(self, x: torch.Tensor, t: float | torch.Tensor, condition: Mapping[str, Any], *, reference: bool) -> torch.Tensor:
        """Call the upstream decoder, optionally through a frozen reference."""

        if x.ndim != 3 or x.shape[-1] != LATENT_CHANNELS:
            raise ValueError("ACE-Step latent must have shape [B,T,64]")
        batch = x.shape[0]
        compute_dtype = self._compute_dtype
        model_input = x.to(self.device, dtype=compute_dtype)
        enc = self._expand_condition_tensor(condition["encoder_hidden_states"], batch, dtype=compute_dtype)
        enc_mask = self._expand_condition_tensor(condition["encoder_attention_mask"], batch)
        context = self._expand_condition_tensor(condition["context_latents"], batch, dtype=compute_dtype)
        attention = self._expand_condition_tensor(condition["attention_mask"], batch)
        if isinstance(t, torch.Tensor):
            timestep = t.to(device=self.device, dtype=compute_dtype)
            if timestep.ndim == 0:
                timestep = timestep.expand(batch)
            elif tuple(timestep.shape) != (batch,):
                raise ValueError("timestep must be scalar or have shape [B]")
        else:
            timestep = torch.full((batch,), float(t), device=self.device, dtype=compute_dtype)
        kwargs = {
            "hidden_states": model_input,
            "timestep": timestep,
            "timestep_r": timestep,
            "attention_mask": attention,
            "encoder_hidden_states": enc,
            "encoder_attention_mask": enc_mask,
            "context_latents": context,
            "use_cache": False,
        }

        def call(module: nn.Module) -> torch.Tensor:
            output = module(**kwargs)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise TypeError("ACE-Step decoder did not return a velocity tensor")
            return output

        with self._autocast():
            if not reference:
                output = call(self.decoder)
            elif self._reference_mode == "lora":
                from .policy import disable_adapters

                with disable_adapters(self.decoder):
                    output = call(self.decoder)
            elif self._reference_mode == "full":
                try:
                    from torch.func import functional_call
                except ImportError:  # pragma: no cover - old PyTorch fallback
                    from torch.nn.utils.stateless import functional_call
                output = functional_call(
                    self.decoder,
                    self._reference_state_for_device(),
                    args=(),
                    kwargs=kwargs,
                    strict=False,
                )
                if isinstance(output, (tuple, list)):
                    output = output[0]
            else:
                raise RuntimeError("prepare_reference must be called before reference velocity")
        if output.shape != x.shape:
            raise RuntimeError(f"ACE-Step velocity shape {tuple(output.shape)} != latent shape {tuple(x.shape)}")
        return output.to(dtype=x.dtype)

    def velocity(
        self,
        x: torch.Tensor,
        t: float | torch.Tensor,
        condition: Mapping[str, Any],
        *,
        reference: bool = False,
    ) -> torch.Tensor:
        """Return the direct ACE flow velocity with CFG intentionally disabled."""

        return self._decoder_call(x, t, condition, reference=reference)

    def prepare_reference(self, mode: str) -> None:
        """Freeze the pre-policy reference used for paired rollouts/KL."""

        normalized = str(mode).strip().lower()
        if normalized not in {"lora", "full"}:
            raise ValueError("reference mode must be 'lora' or 'full'")
        if self._reference_mode is not None:
            if self._reference_mode != normalized:
                raise RuntimeError("reference mode cannot be changed after preparation")
            return
        self._reference_mode = normalized
        if normalized == "full":
            # Full mode intentionally stores the immutable copy on CPU.  The
            # device mirror is lazy and is reused for every reference call;
            # this costs roughly one additional decoder copy in host/device
            # memory, but avoids a second trainable module and its optimizer
            # state.
            self._reference_state = {
                name: value.detach().cpu().clone()
                for name, value in self.decoder.state_dict().items()
            }

    def decode(self, latent: torch.Tensor) -> DecodedAudio:
        """Decode one latent sample to raw 48 kHz stereo audio, unnormalized."""

        if not isinstance(latent, torch.Tensor) or latent.ndim not in {2, 3}:
            raise ValueError("latent must have shape [T,64] or [1,T,64]")
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)
        if latent.shape[0] != 1 or latent.shape[-1] != LATENT_CHANNELS:
            raise ValueError("decode expects exactly one [1,T,64] latent")
        vae_input = latent.to(self.device, dtype=self._load_dtype).transpose(1, 2).contiguous()
        with torch.inference_mode():
            decoded = self.vae.decode(vae_input)
            samples = decoded.sample if hasattr(decoded, "sample") else decoded
        if not isinstance(samples, torch.Tensor) or samples.ndim != 3 or samples.shape[1] != 2:
            raise RuntimeError("ACE-Step VAE decode must return [1,2,samples]")
        # Deliberately do not peak-normalize/clamp: the reward gate must see
        # the native VAE output and reject invalid amplitudes itself.
        array = samples[0].transpose(0, 1).detach().cpu().float().numpy()
        return DecodedAudio(np.asarray(array, dtype=np.float32), SAMPLE_RATE)

    def provenance(self) -> dict[str, Any]:
        """Return reproducibility metadata for the local model boundary."""

        return {
            "backend": "acestep",
            "upstream_dir": str(self.upstream_dir),
            "upstream_commit": self.cfg.upstream_commit,
            "audited_upstream_commit": AUDITED_UPSTREAM_COMMIT,
            "flow_grpo_commit": AUDITED_FLOW_GRPO_COMMIT,
            "sft_config_commit": AUDITED_SFT_CONFIG_COMMIT,
            "vae_config_commit": AUDITED_VAE_CONFIG_COMMIT,
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_name": self.cfg.checkpoint_name,
            "checkpoint_architecture": {
                "hidden_size": self._model_config_json["hidden_size"],
                "num_hidden_layers": self._model_config_json["num_hidden_layers"],
                "in_channels": self._model_config_json["in_channels"],
                "audio_acoustic_hidden_dim": self._model_config_json["audio_acoustic_hidden_dim"],
                "is_turbo": self._model_config_json["is_turbo"],
            },
            "vae_architecture": {
                "sampling_rate": self._vae_config_json["sampling_rate"],
                "audio_channels": self._vae_config_json["audio_channels"],
                "decoder_input_channels": self._vae_config_json["decoder_input_channels"],
                "downsampling_ratios": self._vae_config_json["downsampling_ratios"],
                "downsampling_product": VAE_DOWNSAMPLE,
            },
            "artefact_hashes": self._artefact_hashes,
            "source_code_reviewed": True,
            "local_files_only": True,
            "network_access": False,
            "reference_mode": self._reference_mode,
            "reference_memory": (
                "lora uses one decoder with adapters disabled for reference"
                if self._reference_mode == "lora"
                else "full stores an immutable CPU decoder state and lazily materializes a device copy"
                if self._reference_mode == "full"
                else "not prepared"
            ),
            "guidance_scale": 1.0,
            "sample_rate": SAMPLE_RATE,
            "latent_hz": LATENT_HZ,
            "latent_channels": LATENT_CHANNELS,
        }


__all__ = [
    "AceStepBackend",
    "AUDITED_UPSTREAM_COMMIT",
    "AUDITED_SFT_CONFIG_COMMIT",
    "AUDITED_VAE_CONFIG_COMMIT",
    "AUDITED_FLOW_GRPO_COMMIT",
]
