import os
import argparse
import time

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import dist_utils
from src.data_utils import get_data
from src.owl_pruner import OWLFastOBCPruner


def parse_args():
    parser = argparse.ArgumentParser(description="One-shot pruning OWL pruning.")
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being pruned",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    parser.add_argument(
        "--prunable_modules",
        type=str,
        required=True,
        help="Regex for modules to prune",
    )
    parser.add_argument(
        "--pre_block_modules",
        nargs="+",
        type=str,
        required=True,
        help="Names of modules before transformer blocks",
    )
    parser.add_argument(
        "--block_modules",
        type=str,
        required=True,
        help="Name of transformer modules",
    )
    # Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", required=True, type=int, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", default=None, type=int, help="Length of calibration sequences."
    )
    # Sparsification params
    parser.add_argument("--sparsity", required=True, type=float)
    parser.add_argument("--owl_lambda", nargs="+", required=True, type=float)
    parser.add_argument("--owl_m", nargs="+", required=True, type=float)
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    parser.add_argument("--block_size", type=int, default=128)
    # Save params
    parser.add_argument("--save_dir", type=str, required=True, help="where to save sparse model.")
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed.")
    parser.add_argument(
        "--low_cpu_mem_usage", action="store_true", help="whether to load model with the use of `low_cpu_mem_usage`"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--verbose", action="store_true", help="whether to log progress.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Distributed init
    if dist.is_available():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    # init device
    device = f"cuda:{rank}"
    dtype = getattr(torch, args.dtype)
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        attn_implementation=args.attn_implementation,
    )
    if not args.cpu_offload_modules:
        model = model.to(device)
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path, use_fast=False)
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or model.config.max_position_embeddings
    calibration_data = get_data(
        args.calibration_data, args.calibration_tokens, args.calibration_sequence_length, tokenizer, train=True
    )
    # take slice (if running on multiple workers)
    if dist_utils.is_dist_available_and_initialized():
        num_seq_per_rank = len(calibration_data) // world_size
        calibration_data = calibration_data[rank * num_seq_per_rank : (rank + 1) * num_seq_per_rank]
    calibration_data = [([], {"input_ids": input_ids}) for input_ids in calibration_data]
    dist.barrier()
    # Pruner
    pruner = OWLFastOBCPruner(
        model,
        calibration_data,
        prunable_modules=args.prunable_modules,
        pre_block_modules=args.pre_block_modules,
        block_modules=args.block_modules,
        save_dir=args.save_dir,
        rel_damp=args.rel_damp,
        block_size=args.block_size,
        device=device,
        cpu_offload_modules=args.cpu_offload_modules,
        cpu_offload_activations=args.cpu_offload_activations,
        verbose=args.verbose,
        owl_m=args.owl_m,
        owl_lambda=args.owl_lambda,
    )
    # Prepare save dir
    if dist_utils.is_main():
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(
            {"sparsity": args.sparsity, "owl_m": args.owl_m, "owl_lambda": args.owl_lambda},
            os.path.join(args.save_dir, "metadata.pth"),
        )
    dist.barrier()
    t1 = time.perf_counter()
    pruner.prune(args.sparsity)
    t2 = time.perf_counter()
    dist_utils.print_on_main(f"Pruning took {(t2 - t1)} s.")


if __name__ == "__main__":
    main()
