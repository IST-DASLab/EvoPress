import argparse
import os
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Conversion of weight database to HF checkpoints.")
    # Model params
    parser.add_argument(
        "--awq_checkpoint_dir",
        type=str,
        required=True,
        help="The name or path to AWQ checkpoint directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the AWQ database in EvoPress format",
    )
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    os.makedirs(args.output_dir, exist_ok=True)
    # Load model
    for path in os.listdir(args.awq_checkpoint_dir):
        if path.startswith(args.model_name) and path.endswith("dequantized"):
            print(f"Loading quantized model from {path}.")
            bits = int(re.search("w\d+", path)[0][1:])
            model = AutoModelForCausalLM.from_pretrained(os.path.join(args.awq_checkpoint_dir, path), low_cpu_mem_usage=True, torch_dtype=dtype)
            for i, module in enumerate(model.model.layers):
                os.makedirs(os.path.join(args.output_dir, args.model_name, f"model.layers.{i}"), exist_ok=True)
                torch.save(module.state_dict(), os.path.join(args.output_dir, args.model_name, f"model.layers.{i}", f"{bits}.pth"))

if __name__ == "__main__":
    main()
