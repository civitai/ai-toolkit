import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import AnimaTextConditioner
from diffusers.models import CosmosTransformer3DModel
from safetensors.torch import load_file


DEFAULT_ANIMA_EXTRAS_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"


@dataclass
class PreparedAnimaPaths:
    diffusion_path: str
    extras_path: str
    converted_diffusion: bool
    text_conditioner_path: str | None = None


COSMOS_2_T2I_CONFIG = {
    "in_channels": 16,
    "out_channels": 16,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 28,
    "mlp_ratio": 4.0,
    "text_embed_dim": 1024,
    "adaln_lora_dim": 256,
    "max_size": (128, 240, 240),
    "patch_size": (1, 2, 2),
    "rope_scale": (1.0, 4.0, 4.0),
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
}


COSMOS_2_RENAMES = {
    "t_embedder.1": "time_embed.t_embedder",
    "t_embedding_norm": "time_embed.norm",
    "blocks": "transformer_blocks",
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
    "self_attn": "attn1",
    "cross_attn": "attn2",
    "q_proj": "to_q",
    "k_proj": "to_k",
    "v_proj": "to_v",
    "output_proj": "to_out.0",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
    "x_embedder.proj.1": "patch_embed.proj",
    "final_layer.adaln_modulation.1": "norm_out.linear_1",
    "final_layer.adaln_modulation.2": "norm_out.linear_2",
    "final_layer.linear": "proj_out",
}


COSMOS_2_DROP_KEYS = (
    "accum_video_sample_counter",
    "accum_image_sample_counter",
    "accum_iteration",
    "accum_train_in_hours",
    "pos_embedder.seq",
    "pos_embedder.dim_spatial_range",
    "pos_embedder.dim_temporal_range",
    "_extra_state",
)


def _cache_root() -> Path:
    return Path(os.environ.get("AITK_ANIMA_CONVERSION_CACHE", "~/.cache/ai-toolkit/anima")).expanduser()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(value).stem).strip("-") or "checkpoint"


def has_anima_diffusion_components(path: str) -> bool:
    root = Path(path).expanduser()
    return (
        root.is_dir()
        and (root / "transformer" / "config.json").is_file()
        and (root / "text_conditioner" / "config.json").is_file()
    )


def has_anima_text_conditioner(path: str) -> bool:
    root = Path(path).expanduser()
    return root.is_dir() and (root / "text_conditioner" / "config.json").is_file()


def _checkpoint_cache_dir(checkpoint_path: str) -> Path:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    stat = checkpoint.stat()
    identity = _hash(f"{checkpoint}:{stat.st_size}:{stat.st_mtime_ns}")
    return _cache_root() / f"{_safe_stem(checkpoint.name)}-{identity}"


def _resolve_checkpoint(value: str) -> str:
    if value.startswith(("http://", "https://")):
        raise ValueError(
            "Anima checkpoint conversion expects model.name_or_path to be a local .safetensors file. "
            "Download the checkpoint first and pass the local path."
        )

    path = Path(value).expanduser()
    if path.is_dir():
        raise ValueError(
            "Anima checkpoint conversion expects model.name_or_path to point directly to a .safetensors file, "
            f"not a directory: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Anima checkpoint file not found: {path}")
    if path.suffix != ".safetensors":
        raise ValueError(f"Anima checkpoint must be a .safetensors file: {path}")

    return str(path)


def _check_keys(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], label: str) -> None:
    expected = set(model.state_dict().keys())
    actual = set(state_dict.keys())
    missing = expected - actual
    unexpected = actual - expected
    if not missing and not unexpected:
        return

    details = []
    if missing:
        details.append(f"missing {label} keys ({len(missing)}): {sorted(missing)[:20]}")
    if unexpected:
        details.append(f"unexpected {label} keys ({len(unexpected)}): {sorted(unexpected)[:20]}")
    raise ValueError("; ".join(details))


def split_anima_checkpoint(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    transformer_state_dict = {}
    text_conditioner_state_dict = {}
    adapter_prefixes = (
        "net.llm_adapter.",
        "model.diffusion_model.llm_adapter.",
        "diffusion_model.llm_adapter.",
    )

    for key, value in state_dict.items():
        for adapter_prefix in adapter_prefixes:
            if key.startswith(adapter_prefix):
                text_conditioner_state_dict[key.removeprefix(adapter_prefix)] = value
                break
        else:
            transformer_state_dict[key] = value

    return transformer_state_dict, text_conditioner_state_dict


def convert_cosmos_2_transformer(state_dict: dict[str, torch.Tensor]) -> CosmosTransformer3DModel:
    with init_empty_weights():
        transformer = CosmosTransformer3DModel(**COSMOS_2_T2I_CONFIG)

    converted = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("model.diffusion_model.", "diffusion_model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key.removeprefix(prefix)
                break
        for old, new in COSMOS_2_RENAMES.items():
            new_key = new_key.replace(old, new)
        if not any(drop_key in new_key for drop_key in COSMOS_2_DROP_KEYS):
            converted[new_key] = value

    _check_keys(transformer, converted, "transformer")
    transformer.load_state_dict(converted, strict=True, assign=True)
    return transformer


def infer_text_conditioner_config(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    model_dim = state_dict["blocks.0.self_attn.q_proj.weight"].shape[0]
    source_dim = state_dict["blocks.0.cross_attn.k_proj.weight"].shape[1]
    target_vocab_size, target_dim = state_dict["embed.weight"].shape
    attention_head_dim = state_dict["blocks.0.self_attn.q_norm.weight"].shape[0]
    num_layers = 1 + max(int(key.split(".")[1]) for key in state_dict if key.startswith("blocks."))
    return {
        "source_dim": source_dim,
        "target_dim": target_dim,
        "model_dim": model_dim,
        "num_layers": num_layers,
        "num_attention_heads": model_dim // attention_head_dim,
        "target_vocab_size": target_vocab_size,
    }


def convert_text_conditioner(state_dict: dict[str, torch.Tensor]) -> AnimaTextConditioner:
    with init_empty_weights():
        text_conditioner = AnimaTextConditioner(**infer_text_conditioner_config(state_dict))

    _check_keys(text_conditioner, state_dict, "text conditioner")
    text_conditioner.load_state_dict(state_dict, strict=True, assign=True)
    return text_conditioner


def convert_anima_checkpoint(
    checkpoint_path: str,
    output_path: str,
    *,
    dtype: torch.dtype,
    max_shard_size: str = "5GB",
) -> str:
    if has_anima_diffusion_components(output_path):
        return output_path

    transformer_state_dict, text_conditioner_state_dict = split_anima_checkpoint(
        load_file(checkpoint_path, device="cpu")
    )
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    convert_cosmos_2_transformer(transformer_state_dict).to(dtype=dtype).save_pretrained(
        output / "transformer",
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    if text_conditioner_state_dict:
        convert_text_conditioner(text_conditioner_state_dict).to(dtype=dtype).save_pretrained(
            output / "text_conditioner",
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
    else:
        print(
            "Anima checkpoint does not contain text conditioner weights; "
            "loading text_conditioner from extras_name_or_path"
        )
    return str(output)


def prepare_anima_component_paths(
    name_or_path: str,
    extras_name_or_path: str,
    *,
    dtype: torch.dtype,
    max_shard_size: str = "5GB",
) -> PreparedAnimaPaths:
    model_path = os.path.expanduser(str(name_or_path))
    if has_anima_diffusion_components(model_path):
        return PreparedAnimaPaths(model_path, extras_name_or_path, False)

    if model_path.startswith(("http://", "https://")):
        raise ValueError(
            "Anima checkpoint conversion expects model.name_or_path to be a local .safetensors file. "
            "Download the checkpoint first and pass the local path."
        )

    if not Path(model_path).exists() and not model_path.endswith(".safetensors"):
        return PreparedAnimaPaths(model_path, extras_name_or_path, False)

    extras_path = extras_name_or_path
    if not extras_path or extras_path == name_or_path:
        extras_path = DEFAULT_ANIMA_EXTRAS_REPO

    checkpoint_path = _resolve_checkpoint(model_path)
    output_path = _checkpoint_cache_dir(checkpoint_path)
    print(f"Converting Anima checkpoint to Diffusers transformer components: {checkpoint_path} -> {output_path}")
    converted_path = convert_anima_checkpoint(
        checkpoint_path,
        str(output_path),
        dtype=dtype,
        max_shard_size=max_shard_size,
    )

    text_conditioner_path = converted_path if has_anima_text_conditioner(converted_path) else extras_path
    return PreparedAnimaPaths(converted_path, extras_path, True, text_conditioner_path)
