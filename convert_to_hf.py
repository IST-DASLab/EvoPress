import argparse
import os
from typing import Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.model_utils import load_sparse_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Conversion of weight database to HF checkpoints.")
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being pruned",
    )
    parser.add_argument(
        "--compressed_weights_path",
        type=str,
        required=True,
        help="Path to sparse weights",
    )
    parser.add_argument(
        "--compressed_config_path",
        type=str,
        default=None,
        help="Path to sparse config. By default uniform sparsity.",
    )
    parser.add_argument(
        "--default_level",
        type=int,
        default=0,
        help="Default sparsity level.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    # Save params
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to save sparse model",
    )
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, torch_dtype=dtype)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast=args.use_fast_tokenizer
    )
    # Load compressed weights and save
    load_sparse_weights(model, args.compressed_weights_path, args.compressed_config_path, args.default_level)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
