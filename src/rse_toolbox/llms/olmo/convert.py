from typing import Any, Dict
import gc
import shutil

from pathlib import Path
from hf_olmo import OLMoForCausalLM
from transformers import OlmoConfig
from transformers import OlmoForCausalLM as TFOLMoForCausalLM
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
from transformers import AutoTokenizer

import torch

from ...io import read_json, write_json

def _write_tokenizer(
    input_tokenizer_dir: Path,
    output_tokenizer_dir: Path,
    config: OlmoConfig,
    **kwargs,
) -> None:
    print(f"Saving a {GPTNeoXTokenizerFast.__name__} to {output_tokenizer_dir}.")

    base_tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_dir)
    base_tokenizer_obj = base_tokenizer._tokenizer
    
    eos_token_id = config.eos_token_id if config.eos_token_id is not None else base_tokenizer_obj.get_vocab_size() - 1
    pad_token_id = config.pad_token_id if config.pad_token_id is not None else eos_token_id

    tokenizer = GPTNeoXTokenizerFast(
        tokenizer_object=base_tokenizer_obj,
        eos_token=base_tokenizer_obj.decode([eos_token_id], skip_special_tokens=False),
        pad_token=base_tokenizer_obj.decode([pad_token_id], skip_special_tokens=False),
        unk_token=None,
        bos_token=None,
        chat_template=base_tokenizer.chat_template,
        **kwargs,
    )

    tokenizer.save_pretrained(output_tokenizer_dir)

def to_hf(
    input_model_dir: str | Path,
    output_model_dir: str | Path,
    safe_serialization: bool = True,
    fix_eos_token_id=True
) -> None:
    """
    Converts an OLMo model to the new Hugging Face Transformers format,
    which supports transformers >= 4.40.
    
    Original code from https://github.com/allenai/OLMo/blob/main/scripts/convert_olmo_to_hf_new.py

    Parameters
    ----------
    input_model_dir : str | Path
        The directory where the OLMo model is stored.
    output_model_dir : str | Path
        The directory where the new Hugging Face version
        of the model will be stored.
    safe_serialization : bool, optional
        Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
    fix_eos_token_id : bool, optional
        Whether to fix the eos token id from 0 to 50279.
    """
    # Convert string paths to Path objects
    if isinstance(input_model_dir, str):
        input_model_dir = Path(input_model_dir)
    if isinstance(output_model_dir, str):
        output_model_dir = Path(output_model_dir)

    # Create temporary directory
    tmp_model_path = output_model_dir / "tmp"
    tmp_model_path.mkdir(parents=True, exist_ok=True)
    
    # Load the original OLMO config
    olmo_config = read_json(input_model_dir / "config.json")
    
    # Fetch all the parameters
    n_layers = olmo_config["n_layers"]
    n_heads = olmo_config["n_heads"]
    dim = olmo_config["d_model"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    max_position_embeddings = olmo_config["max_sequence_length"]

    vocab_size = olmo_config.get("embedding_size", olmo_config["vocab_size"])

    if olmo_config.get("n_kv_heads", None) is not None:
        num_key_value_heads = olmo_config["n_kv_heads"]  # for GQA / MQA
    elif olmo_config["multi_query_attention"]:  # compatibility with other checkpoints
        num_key_value_heads = 1
    else:
        num_key_value_heads = n_heads

    # TODO: Change print to logger
    print(f"Fetching all parameters from the checkpoint at {input_model_dir}.")
    
    # Load the original model
    olmo = OLMoForCausalLM.from_pretrained(input_model_dir)
    
    # Retrieve the state dictionary
    # Not sharded
    loaded: Dict[str, torch.Tensor] = olmo.model.state_dict()
    
    # Free up memory
    del olmo
    gc.collect()
    
    # Create new sharded model
    # TODO: Change print to logger
    print("Creating hf version of model layers.")
    param_count = 0
    index_dict: Dict[str, Any] = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        output_file = tmp_model_path / filename
        # Unsharded
        # TODO: Layernorm stuff
        # TODO: multi query attention
        fused_dims = [dim, dims_per_head * num_key_value_heads, dims_per_head * num_key_value_heads]
        q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
            loaded[f"transformer.blocks.{layer_i}.att_proj.weight"], fused_dims, dim=0
        )
        up_proj_weight, gate_proj_weight = torch.chunk(
            loaded[f"transformer.blocks.{layer_i}.ff_proj.weight"], 2, dim=0
        )
        o_proj_weight = loaded[
            f"transformer.blocks.{layer_i}.attn_out.weight"
        ]
        down_proj_weight = loaded[f"transformer.blocks.{layer_i}.ff_out.weight"]
        state_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": q_proj_weight,
            f"model.layers.{layer_i}.self_attn.k_proj.weight": k_proj_weight,
            f"model.layers.{layer_i}.self_attn.v_proj.weight": v_proj_weight,
            f"model.layers.{layer_i}.self_attn.o_proj.weight": o_proj_weight,
            f"model.layers.{layer_i}.mlp.gate_proj.weight": gate_proj_weight,
            f"model.layers.{layer_i}.mlp.down_proj.weight": down_proj_weight,
            f"model.layers.{layer_i}.mlp.up_proj.weight": up_proj_weight,
        }

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        # TODO: Change print to logger
        print(f"Saving {filename}.")
        torch.save(state_dict, output_file)
    
    # Create last shard    
    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    output_file = tmp_model_path / filename
    # Unsharded
    # TODO: Deal with weight-tying
    state_dict = {
        "model.embed_tokens.weight": loaded["transformer.wte.weight"],
        "lm_head.weight": loaded["transformer.ff_out.weight"]
        if "transformer.ff_out.weight" in loaded
        else loaded["transformer.wte.weight"],
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    print(f"Saving {filename}.")
    torch.save(state_dict, output_file)
    
    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()
    
    # Create the new config
    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, tmp_model_path / "pytorch_model.bin.index.json")

    if olmo_config.get("mlp_hidden_size", None) is not None:
        intermediate_size = olmo_config["mlp_hidden_size"] // 2
    else:
        intermediate_size = (dim * olmo_config["mlp_ratio"]) // 2

    if fix_eos_token_id and olmo_config["eos_token_id"] == 0:
        # TODO: Change print to logger
        # Fixing a bug in OLMo where eos token id was incorrectly set
        print("Changing eos_token_id from 0 to 50279.")
        olmo_config["eos_token_id"] = 50279

    config = OlmoConfig(
        vocab_size=vocab_size,
        hidden_size=dim,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=olmo_config["pad_token_id"],
        bos_token_id=None,
        eos_token_id=olmo_config["eos_token_id"],
        tie_word_embeddings=olmo_config["weight_tying"],
        rope_theta=base,
        clip_qkv=olmo_config.get("clip_qkv"),
    )
    config.save_pretrained(tmp_model_path)
    
    # Write the tokenizer
    _write_tokenizer(input_model_dir, output_model_dir, config)
    
    # Loading the new hf version of the model and saving it
    # TODO: Change print to logger
    print("Loading the checkpoint in a OLMo model.")
    model = TFOLMoForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(output_model_dir, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)
    print("Successfully converted the model.")
    
    
def to_gguf() -> None:
    ...