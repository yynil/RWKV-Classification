import os
os.environ["RWKV_JIT_ON"] = '0'
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)
ckpt_file = '/Volumes/TOUROS/models/rwkv/RWKV-4-World-0.4B-v1-20230529-ctx4096.pth'
# model = RWKV(model=ckpt_file, strategy='cuda fp16')

# out, state = model.forward([187, 510, 1563, 310, 247], None)   # use 20B_tokenizer.json
# print(out.detach().cpu().numpy())              
# print(out.shape)
# print(out.detach().cpu().numpy())                   # same result as above
# print(len(state))
# print(state[0].shape)

# tokenizer_file = '/home/yueyulin/pretrained_model/rwkv/rwkv_vocab_v20230424.txt'
# from models import RWKV_TOKENIZER
# tokenizer = RWKV_TOKENIZER(tokenizer_file)
# print(tokenizer)
# str_input = "我在北京等你。"
# inputs = tokenizer.encode(str_input)
# print(inputs)



import argparse
import gc
import json
import os
import re

import torch
from huggingface_hub import hf_hub_download

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, RwkvConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, shard_checkpoint


NUM_HIDDEN_LAYERS_MAPPING = {
    "169M": 12,
    "430M": 24,
    "1B5": 24,
    "3B": 32,
    "7B": 32,
    "14B": 40,
}

HIDEN_SIZE_MAPPING = {
    "169M": 768,
    "430M": 1024,
    "1B5": 2048,
    "3B": 2560,
    "7B": 4096,
    "14B": 5120,
}


def convert_state_dict(state_dict):
    state_dict_keys = list(state_dict.keys())
    for name in state_dict_keys:
        weight = state_dict.pop(name)
        # emb -> embedding
        if name.startswith("emb."):
            name = name.replace("emb.", "embeddings.")
        # ln_0 -> pre_ln (only present at block 0)
        if name.startswith("blocks.0.ln0"):
            name = name.replace("blocks.0.ln0", "blocks.0.pre_ln")
        # att -> attention
        name = re.sub(r"blocks\.(\d+)\.att", r"blocks.\1.attention", name)
        # ffn -> feed_forward
        name = re.sub(r"blocks\.(\d+)\.ffn", r"blocks.\1.feed_forward", name)
        # time_mix_k -> time_mix_key and reshape
        if name.endswith(".time_mix_k"):
            name = name.replace(".time_mix_k", ".time_mix_key")
        # time_mix_v -> time_mix_value and reshape
        if name.endswith(".time_mix_v"):
            name = name.replace(".time_mix_v", ".time_mix_value")
        # time_mix_r -> time_mix_key and reshape
        if name.endswith(".time_mix_r"):
            name = name.replace(".time_mix_r", ".time_mix_receptance")

        if name != "head.weight":
            name = "rwkv." + name

        state_dict[name] = weight
    return state_dict


def convert_rmkv_checkpoint_to_hf_format(
    checkpoint_file, output_dir, size=None, tokenizer_file=None, push_to_hub=False, model_name=None
):
    # 1. If possible, build the tokenizer.
    if tokenizer_file is None:
        print("No `--tokenizer_file` provided, we will use the default tokenizer.")
        vocab_size = 50277
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        vocab_size = len(tokenizer)
    tokenizer.save_pretrained(output_dir)

    # 2. Build the config
    possible_sizes = list(NUM_HIDDEN_LAYERS_MAPPING.keys())
    if size is None:
        # Try to infer size from the checkpoint name
        for candidate in possible_sizes:
            if candidate in checkpoint_file:
                size = candidate
                break
        if size is None:
            raise ValueError("Could not infer the size, please provide it with the `--size` argument.")
    if size not in possible_sizes:
        raise ValueError(f"`size` should be one of {possible_sizes}, got {size}.")

    config = RwkvConfig(
        vocab_size=vocab_size,
        num_hidden_layers=NUM_HIDDEN_LAYERS_MAPPING[size],
        hidden_size=HIDEN_SIZE_MAPPING[size],
    )
    config.save_pretrained(output_dir)

    # 3. Download model file then convert state_dict
    model_file = checkpoint_file
    state_dict = torch.load(model_file, map_location="cpu")
    state_dict = convert_state_dict(state_dict)

    # 4. Split in shards and save
    shards, index = shard_checkpoint(state_dict)
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(output_dir, shard_file))

    if index is not None:
        save_index_file = os.path.join(output_dir, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

        # 5. Clean up shards (for some reason the file PyTorch saves take the same space as the whole state_dict
        print(
            "Cleaning up shards. This may error with an OOM error, it this is the case don't worry you still have converted the model."
        )
        shard_files = list(shards.keys())

        del state_dict
        del shards
        gc.collect()

        for shard_file in shard_files:
            state_dict = torch.load(os.path.join(output_dir, shard_file))
            torch.save({k: v.cpu().clone() for k, v in state_dict.items()}, os.path.join(output_dir, shard_file))

    del state_dict
    gc.collect()
output_dir = '/Volumes/TOUROS/models/rwkv/raven-0.4b-world'
convert_rmkv_checkpoint_to_hf_format(ckpt_file,output_dir,tokenizer_file=None,size='430M')