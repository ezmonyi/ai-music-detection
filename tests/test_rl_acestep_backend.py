from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="Install the optional rl extra")
from torch import nn

from music_detector.rl.acestep_backend import (
    AceStepBackend,
    _unpack_loading_info,
    _validate_sft_config,
    _validate_vae_config,
)
from music_detector.rl.config import ModelConfig
from music_detector.rl.policy import LoRALinear, configure_policy


SFT_CONFIG = {
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
    "is_turbo": False,
    "architectures": ["AceStepConditionGenerationModel"],
    "num_audio_decoder_hidden_layers": 24,
}


class FakeDecoder(nn.Module):
    """Small module with the same keyword boundary as the ACE decoder."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.context_proj = nn.Linear(128, 64)
        self.encoder_proj = nn.Linear(8, 64)

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        timestep_r,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
        use_cache=False,
    ):
        del timestep_r, attention_mask, encoder_attention_mask, use_cache
        encoder = self.encoder_proj(encoder_hidden_states.mean(dim=1)).unsqueeze(1)
        time = timestep.reshape(-1, 1, 1).to(hidden_states.dtype)
        return (self.q_proj(hidden_states) + self.context_proj(context_latents) + encoder + time,)


class FakeTokenizer:
    def __call__(self, prompt, **kwargs):
        del kwargs
        # Distinguish the two prompt branches while keeping the fake encoder tiny.
        length = 3 if "# Instruction" in prompt else 2
        ids = torch.arange(length, dtype=torch.long).unsqueeze(0)
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))


class FakeTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)

    def forward(self, input_ids):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    def embed_tokens(self, input_ids):
        return self.embedding(input_ids)


class FakeConditionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.null_condition_emb = nn.Parameter(torch.zeros(1, 1, 8))
        self.calls = []

    @torch.no_grad()
    def prepare_condition(self, **kwargs):
        self.calls.append(kwargs)
        text_hidden = kwargs["text_hidden_states"]
        source = kwargs["src_latents"]
        masks = kwargs["chunk_masks"].to(source.dtype)
        return text_hidden, kwargs["text_attention_mask"], torch.cat((source, masks), dim=-1)


class FakeVAE(nn.Module):
    def decode(self, latent):
        del latent
        # Deliberately outside [-1, 1]: decode must preserve native VAE values.
        return SimpleNamespace(sample=torch.full((1, 2, 5), 2.5))


def _backend(decoder=None):
    backend = object.__new__(AceStepBackend)
    backend.cfg = ModelConfig(
        backend="acestep",
        device="cpu",
        precision="float32",
        gradient_checkpointing=False,
    )
    backend.device = torch.device("cpu")
    backend.decoder = decoder or FakeDecoder()
    backend._reference_mode = None
    backend._reference_state = None
    backend._reference_state_device = None
    return backend


def _condition():
    return {
        "encoder_hidden_states": torch.randn(1, 3, 8),
        "encoder_attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "context_latents": torch.randn(1, 4, 128),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
    }


def test_sft_architecture_guard_accepts_2b_non_turbo_and_rejects_turbo():
    _validate_sft_config(SFT_CONFIG)
    turbo = dict(SFT_CONFIG, is_turbo=True)
    with pytest.raises(ValueError, match="non-Turbo"):
        _validate_sft_config(turbo)
    wrong_depth = dict(SFT_CONFIG, num_audio_decoder_hidden_layers=8)
    with pytest.raises(ValueError, match="24 hidden layers"):
        _validate_sft_config(wrong_depth)


def test_vae_contract_guard_accepts_native_48khz_25hz_layout():
    config = {
        "sampling_rate": 48_000,
        "audio_channels": 2,
        "decoder_input_channels": 64,
        "downsampling_ratios": [2, 4, 4, 6, 10],
    }
    _validate_vae_config(config)
    with pytest.raises(ValueError, match="downsampling product"):
        _validate_vae_config(dict(config, downsampling_ratios=[2, 4, 4, 8, 8]))


def test_condition_matches_upstream_prompt_metadata_and_stays_backward_safe():
    backend = _backend()
    backend.silence_latent = torch.zeros(1, 750, 64)
    backend.text_tokenizer = FakeTokenizer()
    backend.text_encoder = FakeTextEncoder()
    backend.model = FakeConditionModel()
    record = {
        "caption": "warm analog synth",
        "lyrics": "hello",
        "language": "zh",
        "duration_s": 0.2,
        "metas": {"bpm": 120, "time_signature": "4/4", "key": "C major"},
    }

    condition = backend.condition(record)
    assert backend.latent_shape(condition) == (1, 128, 64)
    assert not condition["encoder_hidden_states"].is_inference()
    call = backend.model.calls[0]
    assert "- bpm: 120\n- timesignature: 4/4\n- keyscale: C major\n- duration: 0 seconds\n" in backend._text_prompt(record)
    assert "# Languages\nzh\n\n# Lyric\nhello<|endoftext|>" in backend._lyric_prompt(record)
    assert call["is_covers"].dtype == torch.bool

    # A trainable decoder must be able to backpropagate after consuming the
    # cached frozen condition; inference-mode tensors would fail here.
    x = torch.randn(backend.latent_shape(condition), requires_grad=True)
    loss = backend.velocity(x, 0.5, condition).square().mean()
    loss.backward()
    assert x.grad is not None
    assert any(parameter.grad is not None for parameter in backend.decoder.parameters())


def test_lora_reference_disables_adapters_and_restores_state():
    backend = _backend()
    condition = _condition()
    backend.prepare_reference("lora")
    configure_policy(backend.decoder, mode="lora", rank=2, alpha=2, target_modules=["q_proj"])
    adapter = next(module for module in backend.decoder.modules() if isinstance(module, LoRALinear))
    with torch.no_grad():
        adapter.lora_B.fill_(0.1)
    x = torch.randn(1, 4, 64)
    current = backend.velocity(x, 0.5, condition)
    reference = backend.velocity(x, 0.5, condition, reference=True)
    assert not torch.allclose(current, reference)
    assert adapter.adapter_enabled


def test_full_reference_is_immutable_and_functional_call_is_differentiable():
    backend = _backend()
    condition = _condition()
    backend.prepare_reference("full")
    x = torch.randn(1, 4, 64)
    reference_before = backend.velocity(x, 0.5, condition, reference=True)
    with torch.no_grad():
        backend.decoder.q_proj.weight.add_(1.0)
    current = backend.velocity(x, 0.5, condition)
    reference_after = backend.velocity(x, 0.5, condition, reference=True)
    assert not torch.allclose(current, reference_before)
    torch.testing.assert_close(reference_after, reference_before)
    assert all(value.device.type == "cpu" for value in backend._reference_state.values())


def test_full_reference_casts_bf16_snapshot_to_fp32_decoder_without_drift():
    # setup_policy captures the reference before converting full mode to an
    # fp32 optimizer master.  The CPU mock exercises that exact transition.
    backend = _backend(FakeDecoder().bfloat16())
    condition = _condition()
    backend.prepare_reference("full")
    backend.decoder.float()
    x = torch.randn(1, 4, 64)
    current = backend.velocity(x, 0.5, condition)
    reference = backend.velocity(x, 0.5, condition, reference=True)
    torch.testing.assert_close(current, reference)


def test_loading_info_guard_fails_closed_on_partial_or_wrong_loader_results():
    with pytest.raises(RuntimeError, match="incompatible weights"):
        _unpack_loading_info((object(), {"missing_keys": ["decoder.q_proj.weight"]}), component="decoder")
    with pytest.raises(RuntimeError, match="did not return loading information"):
        _unpack_loading_info(object(), component="decoder")


def test_decode_preserves_raw_native_vae_output_and_contract():
    backend = _backend()
    backend.vae = FakeVAE()
    decoded = backend.decode(torch.zeros(4, 64))
    assert decoded.samples.shape == (5, 2)
    assert decoded.samples.dtype.name == "float32"
    assert decoded.sample_rate == 48_000
    assert float(decoded.samples.max()) == pytest.approx(2.5)


def test_metadata_string_is_preserved_and_language_does_not_invent_english():
    record = {"caption": "x", "metas": "raw metadata\n", "lyrics": ""}
    assert AceStepBackend._metadata_text(record) == "raw metadata\n"
    assert "\nunknown\n" in AceStepBackend._lyric_prompt(record)
