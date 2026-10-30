import pytest
torch = pytest.importorskip("torch", reason="Install the optional rl extra")
from torch import nn

from music_detector.rl.policy import (
    LoRALinear,
    configure_policy,
    disable_adapters,
    export_adapter_state,
    load_adapter_state,
    named_trainable_parameters,
    trainable_parameters,
)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(4, 3),
                "v_proj": nn.Linear(3, 2),
            }
        )
        self.mlp = nn.Linear(2, 2)

    def forward(self, x):
        x = self.attn["q_proj"](x)
        return self.mlp(self.attn["v_proj"](x))


def test_lora_zero_init_preserves_base_and_freezes_original_weights():
    decoder = TinyDecoder()
    original = {name: parameter.detach().clone() for name, parameter in decoder.named_parameters()}
    x = torch.randn(2, 4)
    before = decoder(x)
    metadata = configure_policy(decoder, mode="lora", rank=2, alpha=4, target_modules="attention")
    after = decoder(x)
    torch.testing.assert_close(after, before)
    assert metadata["target_modules"] == ["attn.q_proj", "attn.v_proj"]
    assert all(not parameter.requires_grad for name, parameter in decoder.named_parameters() if ".base." in name)
    assert all(torch.count_nonzero(module.lora_B) == 0 for module in decoder.modules() if isinstance(module, LoRALinear))
    # Untargeted MLP parameters also remain bit-identical and frozen.
    for name, parameter in decoder.named_parameters():
        if name.endswith(".base.weight") or name.endswith(".base.bias") or name.startswith("mlp."):
            source_name = name.replace(".base", "")
            torch.testing.assert_close(parameter, original[source_name])


def test_lora_trainable_parameters_are_only_adapters_and_context_restores():
    decoder = TinyDecoder()
    configure_policy(decoder, rank=2, alpha=2, target_modules=["q_proj"])
    names = [name for name, _ in named_trainable_parameters(decoder)]
    assert names == ["attn.q_proj.lora_A", "attn.q_proj.lora_B"]
    assert trainable_parameters(decoder) == [parameter for _, parameter in named_trainable_parameters(decoder)]
    x = torch.randn(1, 4)
    with pytest.raises(RuntimeError):
        with disable_adapters(decoder):
            disabled = decoder(x)
            assert all(not module.adapter_enabled for module in decoder.modules() if isinstance(module, LoRALinear))
            raise RuntimeError("restore test")
    assert all(module.adapter_enabled for module in decoder.modules() if isinstance(module, LoRALinear))
    torch.testing.assert_close(decoder(x), disabled)  # B starts zero, so outputs coincide initially.


def test_adapter_state_round_trip_and_strict_name_config_checks():
    source = TinyDecoder()
    configure_policy(source, rank=2, alpha=4, target_modules="all-linear")
    with torch.no_grad():
        for module in source.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.fill_(0.2)
    checkpoint = export_adapter_state(source)
    target = TinyDecoder()
    configure_policy(target, rank=2, alpha=4, target_modules="all-linear")
    load_adapter_state(target, checkpoint)
    for key, value in checkpoint["state_dict"].items():
        found = dict(target.named_parameters())[key]
        torch.testing.assert_close(found, value)

    mismatched = TinyDecoder()
    configure_policy(mismatched, rank=1, alpha=4, target_modules="all-linear")
    with pytest.raises(ValueError, match="config mismatch"):
        load_adapter_state(mismatched, checkpoint)


def test_full_decoder_mode_trains_every_decoder_parameter():
    decoder = TinyDecoder()
    metadata = configure_policy(decoder, mode="full")
    assert metadata["mode"] == "full"
    assert all(parameter.requires_grad for parameter in decoder.parameters())
    assert all(not name.startswith("vae") for name, _ in named_trainable_parameters(decoder))
