"""Native, dependency-free LoRA and full-decoder policy modes.

The ACE-Step RL path only needs a small adaptation surface around an existing
decoder.  This module intentionally uses :class:`torch.nn.Linear` and does
not import PEFT.  In LoRA mode the decoder's original parameters are frozen
and selected linears receive a low-rank residual; in full mode every decoder
parameter is trainable.  A caller should pass the decoder itself (not its VAE
or text encoder), which keeps those conditioning components outside the RL
optimizer by construction.
"""

from __future__ import annotations

import contextlib
import fnmatch
import math
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


_SCHEMA_VERSION = 1
_PROFILE_ALIASES = {
    "all",
    "all_linear",
    "all-linears",
    "all-linear",
    "linear",
    "linears",
    "alllinear",
}
_ATTENTION_ALIASES = {"attention", "attn", "attention-only", "attention_only"}
_ATTENTION_TOKENS = (
    "attn",
    "attention",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "query",
    "key",
    "value",
)


class LoRALinear(nn.Module):
    """A frozen :class:`~torch.nn.Linear` plus a trainable low-rank residual.

    ``lora_A`` is initialised with Kaiming uniform noise and ``lora_B`` is
    exactly zero, so wrapping a layer initially preserves the base model's
    outputs.  Adapter parameters use float32 even when the base layer is
    bfloat16/float16; the residual is cast back to the base output dtype after
    the matrix products.  The explicit cast is differentiable and avoids
    low-precision AdamW updates for the small trainable matrices.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("LoRALinear can wrap only torch.nn.Linear modules")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(float(alpha)) or float(alpha) <= 0:
            raise ValueError("alpha must be a positive finite number")
        if not math.isfinite(float(dropout)) or float(dropout) != 0.0:
            raise ValueError("LoRA dropout must be exactly 0 for rollout/replay consistency")

        self.base = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        # Keep a real Dropout module for compatibility with code that inspects
        # ``adapter.dropout.p``; it is intentionally p=0 in every policy
        # configuration so rollout and replay execute the same function.
        self.dropout = nn.Dropout(p=0.0)
        self.lora_dropout = 0.0
        self.adapter_enabled = True

        # Keep the adapters in float32 irrespective of a bf16/fp16 base.  A
        # CPU toy decoder may be float64; float32 still gives deterministic
        # and useful adapter training, while the result is cast to base dtype.
        adapter_device = linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, device=adapter_device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=adapter_device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # The base module's parameters remain registered and in the original
        # objects, but are not optimizer candidates in LoRA mode.
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the original weight for modules that inspect ``.weight``."""

        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    # Common naming aliases (properties avoid duplicate state_dict entries).
    @property
    def lora_down(self) -> nn.Parameter:
        return self.lora_A

    @property
    def lora_up(self) -> nn.Parameter:
        return self.lora_B

    @property
    def device(self) -> torch.device:
        return self.base.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base.weight.dtype

    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    @property
    def active(self) -> bool:
        return self.adapter_enabled

    @active.setter
    def active(self, value: bool) -> None:
        self.adapter_enabled = bool(value)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_output = self.base(input)
        if not self.adapter_enabled:
            return base_output
        # Both casts below are intentional.  The first keeps matmul and
        # adapter gradients in fp32; the second keeps decoder residual dtype
        # and downstream autocast behavior identical to the base path.
        adapter_input = self.dropout(input.to(dtype=self.lora_A.dtype))
        residual = F.linear(F.linear(adapter_input, self.lora_A), self.lora_B)
        residual = residual * self.scaling
        return base_output + residual.to(dtype=base_output.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha:g}, dropout={self.dropout.p:g}, "
            f"enabled={self.adapter_enabled}"
        )


@dataclass
class PolicyMetadata(Mapping[str, Any]):
    """Serializable policy configuration and trainable-parameter statistics."""

    mode: str
    rank: Optional[int]
    alpha: Optional[float]
    dropout: float
    target_profile: str
    target_modules: Tuple[str, ...]
    trainable_param_names: Tuple[str, ...]
    frozen_param_names: Tuple[str, ...]
    trainable_parameter_count: int
    frozen_parameter_count: int
    total_parameter_count: int
    base_parameter_count: int
    adapter_parameter_count: int
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def trainable_params(self) -> int:
        return self.trainable_parameter_count

    @property
    def frozen_params(self) -> int:
        return self.frozen_parameter_count

    def as_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Dataclass ``asdict`` recursively copies tuples but leaves arbitrary
        # config mappings as ordinary values; normalize names for JSON use.
        result["target_modules"] = list(self.target_modules)
        result["trainable_param_names"] = list(self.trainable_param_names)
        result["frozen_param_names"] = list(self.frozen_param_names)
        result["config"] = dict(self.config)
        # Friendly aliases used by checkpoint manifests and callers.
        result["trainable_params"] = result["trainable_parameter_count"]
        result["trainable_parameters"] = result["trainable_parameter_count"]
        result["frozen_params"] = result["frozen_parameter_count"]
        result["frozen_parameters"] = result["frozen_parameter_count"]
        result["total_params"] = result["total_parameter_count"]
        return result

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


def _normalize_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError("mode must be 'lora' or 'full'")
    normalized = mode.strip().lower().replace("-", "_")
    if normalized in {"lora", "low_rank", "adapter"}:
        return "lora"
    if normalized in {"full", "full_decoder", "decoder"}:
        return "full"
    raise ValueError("mode must be 'lora' or 'full'")


def _module_by_path(module: nn.Module, path: str) -> nn.Module:
    current = module
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, (nn.Sequential, nn.ModuleList)) and part.isdigit():
            current = current[int(part)]
        elif isinstance(current, nn.ModuleDict) and part in current:
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def _set_module_by_path(module: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = _module_by_path(module, parent_path)
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and leaf.isdigit():
        parent[int(leaf)] = replacement
    elif isinstance(parent, nn.ModuleDict):
        parent[leaf] = replacement
    else:
        setattr(parent, leaf, replacement)


def _is_attention_name(name: str) -> bool:
    lower = name.lower()
    # Match path components instead of arbitrary substrings such as an
    # unrelated "attn_cache" parameter.
    components = re.split(r"[._/]", lower)
    return any(token in components or token in lower for token in _ATTENTION_TOKENS)


def _resolve_targets(
    decoder: nn.Module,
    target_modules: Optional[Sequence[str] | str],
    profile: Optional[str],
) -> Tuple[str, ...]:
    linear_names = tuple(name for name, child in decoder.named_modules() if name and isinstance(child, nn.Linear))
    if not linear_names:
        raise ValueError("decoder contains no torch.nn.Linear modules")

    requested: Optional[Sequence[str] | str] = target_modules if target_modules is not None else profile
    if requested is None:
        requested = "attention"
    if isinstance(requested, str):
        token = requested.strip().lower().replace(" ", "_")
        if token in _PROFILE_ALIASES:
            selected = linear_names
        elif token in _ATTENTION_ALIASES:
            selected = tuple(name for name in linear_names if _is_attention_name(name))
        else:
            requested = (requested,)
            selected = tuple(name for name in linear_names if _match_target(name, requested))
    else:
        requested = tuple(str(item) for item in requested)
        if not requested:
            raise ValueError("target_modules cannot be empty")
        selected = tuple(name for name in linear_names if _match_target(name, requested))
    if not selected:
        raise ValueError(
            "no decoder linears matched target_modules/profile; "
            f"available paths: {', '.join(linear_names)}"
        )
    return selected


def _match_target(name: str, patterns: Sequence[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    for pattern in patterns:
        pattern = str(pattern)
        if pattern == name or pattern == leaf or name.endswith("." + pattern) or name.startswith(pattern + "."):
            return True
        # Shell wildcards are deliberately limited to explicit patterns;
        # plain names above cannot accidentally match an unrelated path.
        if any(mark in pattern for mark in "*?[") and fnmatch.fnmatchcase(name, pattern):
            return True
    return False


def _iter_lora(decoder: nn.Module) -> Iterator[Tuple[str, LoRALinear]]:
    for name, child in decoder.named_modules():
        if isinstance(child, LoRALinear):
            yield name, child


def named_trainable_parameters(decoder: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
    """Yield only parameters with ``requires_grad=True`` in stable name order."""

    yield from ((name, parameter) for name, parameter in decoder.named_parameters() if parameter.requires_grad)


def trainable_parameters(decoder: nn.Module) -> List[nn.Parameter]:
    """Return optimizer-ready parameters, excluding all frozen base weights."""

    return [parameter for _, parameter in named_trainable_parameters(decoder)]


def policy_statistics(decoder: nn.Module) -> Dict[str, Any]:
    """Return lightweight live trainability statistics for a configured decoder."""

    named = list(decoder.named_parameters())
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    adapters = [(name, parameter) for name, parameter in named if ".lora_A" in name or ".lora_B" in name]
    return {
        "trainable_param_names": [name for name, _ in trainable],
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "total_parameter_count": sum(parameter.numel() for _, parameter in named),
        "adapter_parameter_count": sum(parameter.numel() for _, parameter in adapters),
        "frozen_parameter_count": sum(parameter.numel() for name, parameter in named if not parameter.requires_grad),
    }


def configure_policy(
    decoder: nn.Module,
    mode: str = "lora",
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[Sequence[str] | str] = None,
    profile: Optional[str] = None,
    *,
    dropout: float = 0.0,
) -> PolicyMetadata:
    """Configure decoder trainability in-place and return metadata.

    ``mode='lora'`` wraps selected linears once and freezes every original
    decoder parameter.  ``mode='full'`` leaves the module topology unchanged
    and enables all decoder parameters.  Reconfiguring a decoder with already
    wrapped linears raises rather than nesting adapters accidentally.
    """

    if not isinstance(decoder, nn.Module):
        raise TypeError("decoder must be a torch.nn.Module")
    normalized_mode = _normalize_mode(mode)
    if not math.isfinite(float(dropout)) or float(dropout) != 0.0:
        raise ValueError("dropout must be exactly 0 for rollout/replay consistency")

    # A configured decoder should not carry stale gradients or accidentally
    # combine old LoRA modules with a new mode.
    existing = list(_iter_lora(decoder))
    if existing:
        raise ValueError("decoder already contains LoRA adapters; configure it only once")

    if normalized_mode == "full":
        for parameter in decoder.parameters():
            parameter.requires_grad_(True)
        names = tuple(name for name, child in decoder.named_modules() if name and isinstance(child, nn.Linear))
        stats = policy_statistics(decoder)
        return PolicyMetadata(
            mode="full",
            rank=None,
            alpha=None,
            dropout=0.0,
            target_profile="full_decoder",
            target_modules=names,
            trainable_param_names=tuple(stats["trainable_param_names"]),
            frozen_param_names=tuple(name for name, parameter in decoder.named_parameters() if not parameter.requires_grad),
            trainable_parameter_count=stats["trainable_parameter_count"],
            frozen_parameter_count=stats["frozen_parameter_count"],
            total_parameter_count=stats["total_parameter_count"],
            base_parameter_count=stats["total_parameter_count"],
            adapter_parameter_count=0,
            config={"mode": "full", "target_profile": "full_decoder"},
        )

    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("alpha must be a positive finite number")
    selected = _resolve_targets(decoder, target_modules, profile)

    # Freeze all original decoder parameters first.  Parent modules created by
    # a larger pipeline are intentionally out of scope because only decoder is
    # passed to this function.
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    for name in selected:
        original = _module_by_path(decoder, name)
        if not isinstance(original, nn.Linear):
            raise ValueError(f"target path {name!r} is not a torch.nn.Linear")
        _set_module_by_path(decoder, name, LoRALinear(original, rank=rank, alpha=alpha, dropout=0.0))

    stats = policy_statistics(decoder)
    # Every parameter that existed before wrapping is still a base parameter;
    # include non-targeted linears and decoder blocks as well as wrapped
    # ``*.base.*`` entries.  The adapter count is disjoint from this total.
    base_count = stats["total_parameter_count"] - stats["adapter_parameter_count"]
    config = {
        "mode": "lora",
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": 0.0,
        "target_profile": profile or (target_modules if isinstance(target_modules, str) else "explicit"),
        "target_modules": list(selected),
    }
    return PolicyMetadata(
        mode="lora",
        rank=int(rank),
        alpha=float(alpha),
        dropout=0.0,
        target_profile=str(config["target_profile"]),
        target_modules=tuple(selected),
        trainable_param_names=tuple(stats["trainable_param_names"]),
        frozen_param_names=tuple(name for name, parameter in decoder.named_parameters() if not parameter.requires_grad),
        trainable_parameter_count=stats["trainable_parameter_count"],
        frozen_parameter_count=stats["frozen_parameter_count"],
        total_parameter_count=stats["total_parameter_count"],
        base_parameter_count=base_count,
        adapter_parameter_count=stats["adapter_parameter_count"],
        config=config,
    )


@contextlib.contextmanager
def disable_adapters(decoder: nn.Module) -> Iterator[nn.Module]:
    """Temporarily disable every LoRA residual and restore prior state."""

    adapters = list(_iter_lora(decoder))
    previous = [adapter.adapter_enabled for _, adapter in adapters]
    try:
        for _, adapter in adapters:
            adapter.disable_adapter()
        yield decoder
    finally:
        for (_, adapter), was_enabled in zip(adapters, previous):
            adapter.adapter_enabled = was_enabled


def _configuration_from_decoder(decoder: nn.Module) -> Dict[str, Any]:
    adapters = list(_iter_lora(decoder))
    if not adapters:
        return {"mode": "full", "target_modules": []}
    ranks = {adapter.rank for _, adapter in adapters}
    alphas = {adapter.alpha for _, adapter in adapters}
    if len(ranks) != 1 or len(alphas) != 1:
        raise ValueError("decoder has inconsistent LoRA adapter configuration")
    return {
        "mode": "lora",
        "rank": next(iter(ranks)),
        "alpha": next(iter(alphas)),
        "dropout": 0.0,
        "target_modules": [name for name, _ in adapters],
    }


def export_adapter_state(decoder: nn.Module) -> Dict[str, Any]:
    """Export adapters (or full decoder trainables) without base weights."""

    config = _configuration_from_decoder(decoder)
    state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    if config["mode"] == "lora":
        for name, adapter in _iter_lora(decoder):
            state[f"{name}.lora_A"] = adapter.lora_A.detach().cpu().clone()
            state[f"{name}.lora_B"] = adapter.lora_B.detach().cpu().clone()
    else:
        # Full decoder mode has no small adapter state.  Export trainable
        # decoder tensors only; VAE/text modules were never passed in.
        for name, parameter in decoder.named_parameters():
            if parameter.requires_grad:
                state[name] = parameter.detach().cpu().clone()
    return {
        "schema_version": _SCHEMA_VERSION,
        "config": config,
        "state_dict": state,
        # Keep this alias for simple checkpoint consumers while retaining an
        # explicit state_dict key for torch-style loaders.
        "state": state,
    }


def _state_mapping(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if "state_dict" in payload:
        mapping = payload["state_dict"]
    elif "state" in payload:
        mapping = payload["state"]
    else:
        mapping = payload
    if not isinstance(mapping, Mapping):
        raise TypeError("adapter checkpoint state_dict must be a mapping")
    return mapping


def load_adapter_state(
    decoder: nn.Module,
    payload: Mapping[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Load an exported adapter state, checking names/configuration strictly."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping returned by export_adapter_state")
    expected_config = _configuration_from_decoder(decoder)
    supplied_config = payload.get("config", {})
    if strict:
        if payload.get("schema_version", _SCHEMA_VERSION) != _SCHEMA_VERSION:
            raise ValueError("unsupported adapter checkpoint schema version")
        # Compare only fields that define tensor compatibility.  target names
        # are order-insensitive for a human-authored checkpoint, while state
        # key equality below remains exact.
        for key in ("mode", "rank", "alpha", "dropout"):
            if key in supplied_config and supplied_config[key] != expected_config.get(key):
                raise ValueError(
                    f"adapter config mismatch for {key}: checkpoint={supplied_config[key]!r}, "
                    f"decoder={expected_config.get(key)!r}"
                )
        supplied_targets = set(supplied_config.get("target_modules", []))
        expected_targets = set(expected_config.get("target_modules", []))
        if supplied_targets and supplied_targets != expected_targets:
            raise ValueError("adapter target module names do not match decoder")

    incoming = _state_mapping(payload)
    expected: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    if expected_config["mode"] == "lora":
        for name, adapter in _iter_lora(decoder):
            expected[f"{name}.lora_A"] = adapter.lora_A
            expected[f"{name}.lora_B"] = adapter.lora_B
    else:
        for name, parameter in decoder.named_parameters():
            if parameter.requires_grad:
                expected[name] = parameter
    expected_keys = set(expected)
    incoming_keys = set(incoming)
    if strict and incoming_keys != expected_keys:
        missing = sorted(expected_keys - incoming_keys)
        unexpected = sorted(incoming_keys - expected_keys)
        raise ValueError(f"adapter state names mismatch (missing={missing}, unexpected={unexpected})")
    with torch.no_grad():
        for name, parameter in expected.items():
            if name not in incoming:
                continue
            tensor = incoming[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"checkpoint value for {name} is not a tensor")
            if strict and tuple(tensor.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"adapter tensor shape mismatch for {name}: checkpoint={tuple(tensor.shape)}, "
                    f"decoder={tuple(parameter.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


# Concise aliases useful to callers that prefer "policy" terminology.
export_policy_state = export_adapter_state
load_policy_state = load_adapter_state
get_trainable_parameters = trainable_parameters
named_parameters_for_optim = named_trainable_parameters


__all__ = [
    "LoRALinear",
    "PolicyMetadata",
    "configure_policy",
    "disable_adapters",
    "named_trainable_parameters",
    "trainable_parameters",
    "policy_statistics",
    "export_adapter_state",
    "load_adapter_state",
    "export_policy_state",
    "load_policy_state",
    "get_trainable_parameters",
    "named_parameters_for_optim",
]
