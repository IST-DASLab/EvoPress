import argparse
import random
import copy
import os
from tqdm import trange
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.data_utils import get_data
from src.common_utils import fix_seed
from src.metrics import compute_perplexity, compute_kl_div


def load_layers(model: AutoModelForCausalLM, layer_names: List[str], new_state: List[int], sparse_weights_path: str):
    assert hasattr(model, "state")
    for layer_name, new_level, old_level in zip(layer_names, new_state, model.state):
        if new_level != old_level:
            layer = model.get_submodule(layer_name)
            layer.weight.data = torch.load(
                os.path.join(sparse_weights_path, layer_name, f"{new_level}.pth"), map_location=layer.weight.device
            ).to(layer.weight.dtype)
    # Update model state
    model.state = new_state


def compute_fitness(model, data, fitness_fn, target_logits: Optional[torch.Tensor] = None) -> float:
    if fitness_fn == "ppl":
        return compute_perplexity(model, data)
    else:
        return compute_kl_div(model, data, target_logits)


def selection(
    model,
    layer_names,
    sparse_weights_path: str,
    candidates,
    num_survive: int,
    calibration_data,
    num_tokens: int,
    fitness_fn: str = "ppl",
    target_logits: Optional[List[torch.Tensor]] = None,
):
    calibration_minibatch = []
    minibatch_ids = []
    target_logits_minibatch = []
    tokens_used = 0
    while tokens_used < num_tokens:  # generate minibatch with exactly num_tokens tokens
        minibatch_id = random.randint(0, len(calibration_data) - 1)
        if minibatch_id in minibatch_ids:  # avoid duplicates
            continue
        minibatch_ids.append(minibatch_id)
        if tokens_used + calibration_data[minibatch_id].shape[1] > num_tokens:
            calibration_minibatch.append(calibration_data[minibatch_id][:, : num_tokens - tokens_used])
            if fitness_fn == "kl":
                target_logits_minibatch.append(target_logits[minibatch_id][:, : num_tokens - tokens_used])
            tokens_used = num_tokens
        else:
            calibration_minibatch.append(calibration_data[minibatch_id])
            if fitness_fn == "kl":
                target_logits_minibatch.append(target_logits[minibatch_id])
            tokens_used += calibration_data[minibatch_id].shape[1]

    if len(target_logits_minibatch) == 0:
        target_logits_minibatch = None

    fitnesses = []
    for candidate in candidates:
        load_layers(model, layer_names, candidate, sparse_weights_path)
        fitness = compute_fitness(model, calibration_minibatch, fitness_fn, target_logits_minibatch)
        fitnesses.append(fitness)
    # Keep only best
    best_ids = np.argsort(fitnesses)[:num_survive]
    return [candidates[i] for i in best_ids], [fitnesses[i] for i in best_ids]


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="The name or path to the model being pruned",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
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
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["fineweb_edu", "wikitext2", "c4"],
        help="Datasets used for evaluation",
    )
    parser.add_argument("--eval_every", default=1, type=int, help="Eval every # generations.")
    parser.add_argument("--eval_tokens", default=524288, type=int, help="Number of tokens for evaluation.")
    parser.add_argument("--eval_sequence_length", default=None, type=int, help="Length of evaluation sequences.")
    parser.add_argument("--fitness_fn", choices=["ppl", "kl"], default="kl", help="Fitness function.")
    parser.add_argument("--max_level", default=99999, type=int, help="Max admissible level.")
    parser.add_argument(
        "--max_total_deviation",
        default=99999,
        type=int,
        help="Max admissible total deviation (sum of absolute differences to uniform pruning).",
    )
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")
    # Evolutionary Search params
    parser.add_argument("--generations", type=int, required=True, help="Number of generations in evolutionary search")
    parser.add_argument("--offspring", type=int, required=True, help="Number of offspring generated per parent")
    parser.add_argument("--sparse_weights_path", type=str, required=True, help="Path to sparse weights")
    parser.add_argument(
        "--survivors_per_selection",
        type=int,
        nargs="+",
        required=True,
        help="Number of survivors after each stage of selection",
    )
    parser.add_argument(
        "--tokens_per_selection",
        type=int,
        nargs="+",
        required=True,
        help="Number of calibration tokens at each stage of selection",
    )
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument(
        "--memory_efficient", action="store_true", help="Whether to use memory efficient implementation."
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    # Save params
    parser.add_argument(
        "--configuration_name", type=str, default="final_configuration.txt", help="Name of final configuration"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Fix seed
    fix_seed(args.seed)
    # Init W&B logger
    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)
    # init device
    device = f"cuda"
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=None if args.memory_efficient else "auto",
        low_cpu_mem_usage=True,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False  # do not use cache
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast=args.use_fast_tokenizer
    )
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or min(
        model.config.max_position_embeddings, 8192
    )
    calibration_data = get_data(
        args.calibration_data,
        args.calibration_tokens,
        args.calibration_sequence_length,
        tokenizer,
        train=True,
    )
    # Load eval datasets
    args.eval_sequence_length = args.eval_sequence_length or min(model.config.max_position_embeddings, 8192)
    eval_datasets = []
    for eval_dataset_name in args.eval_datasets:
        eval_datasets.append(
            get_data(
                eval_dataset_name,
                args.eval_tokens,  # ignored for WikiText2 and C4
                args.eval_sequence_length,
                tokenizer,
                train=False,
            )
        )
    target_logits = []
    if args.fitness_fn == "kl":
        # Compute target logits (calibration)
        for i in trange(0, len(calibration_data), desc="Computing target logits (calib)", leave=False):
            with torch.no_grad():
                target_logits.append(model(calibration_data[i].to(device)).logits.cpu())

    # Prepare layers and initial state
    layer_names = []
    for layer_name in sorted(os.listdir(args.sparse_weights_path)):
        if os.path.isdir(os.path.join(args.sparse_weights_path, layer_name)):
            layer_names.append(layer_name)
    parent = [0 for _ in layer_names]
    model.state = [None] * len(layer_names)

    train_fitness = float("inf")
    log_dict = {}

    for generation in range(args.generations):
        print(f"Generation {generation + 1}/{args.generations}")
        print(f"Current search point: {parent}")
        print(f"Train fitness: {train_fitness:.2e}")

        load_layers(model, layer_names, parent, args.sparse_weights_path)

        # Evaluate current search point
        if generation % args.eval_every == 0:
            for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
                ppl_eval = compute_perplexity(model, eval_dataset)
                print(f"{eval_dataset_name}: {ppl_eval:.2f}")
                log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval
            ppl_train = compute_perplexity(model, calibration_data)
            print(f"ppl_train: {ppl_train:.2f}")
            log_dict["ppl_train"] = ppl_train
        if args.log_wandb:
            wandb.log(log_dict)

        offspring_list = []

        while len(offspring_list) < args.offspring:
            offspring = copy.deepcopy(parent)
            # mutate offspring
            num_flips = min(random.randint(1, 3), random.randint(1, 3))  # bias towards lower values
            for _ in range(num_flips):
                # positions where sparsity can be decreased
                while True:
                    decr_id = random.randint(0, len(offspring) - 1)
                    layer_name = layer_names[decr_id]
                    level = offspring[decr_id]
                    if abs(level - 1) > args.max_level:
                        continue
                    if os.path.exists(os.path.join(args.sparse_weights_path, layer_name, f"{level - 1}.pth")):
                        break
                # positions where sparsity can be increased
                while True:
                    incr_id = random.randint(0, len(offspring) - 1)
                    layer_name = layer_names[incr_id]
                    level = offspring[incr_id]
                    if abs(level + 1) > args.max_level:
                        continue
                    if os.path.exists(os.path.join(args.sparse_weights_path, layer_name, f"{level + 1}.pth")):
                        break
                offspring[decr_id] -= 1
                offspring[incr_id] += 1
            # avoid duplicates
            if offspring in offspring_list:
                continue
            # skip if total deviation exceeds specified threshold
            if sum(map(abs, offspring)) > args.max_total_deviation:
                continue
            offspring_list.append(offspring)

        for num_survive, num_tokens in zip(args.survivors_per_selection, args.tokens_per_selection):
            if num_survive == args.survivors_per_selection[-1]:
                if parent not in offspring_list:  # Elitist EA
                    offspring_list.append(parent)

            offspring_list, train_fitnesses = selection(
                model=model,
                layer_names=layer_names,
                sparse_weights_path=args.sparse_weights_path,
                candidates=offspring_list,
                num_survive=num_survive,
                calibration_data=calibration_data,
                num_tokens=num_tokens,
                fitness_fn=args.fitness_fn,
                target_logits=target_logits,
            )
        # In the end we have lists with a single element (only 1 survivor in last selection step)
        train_fitness = train_fitnesses[0]
        parent = offspring_list[0]
        print(f"Train fitnesses: {train_fitness:.2e}")
        log_dict["train_fitness"] = train_fitness

    # Save final configuration
    with open(os.path.join(args.sparse_weights_path, args.configuration_name), "w") as f:
        f.write("\n".join([f"{layer_name}: {level}" for layer_name, level in zip(layer_names, parent)]))
    # Log final configuration
    print("Final configuration:")
    print(parent)
    # Final evaluation
    for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
        ppl_eval = compute_perplexity(model, eval_dataset)
        print(f"{eval_dataset_name}: {ppl_eval:.2f}")
        log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval
    ppl_train = compute_perplexity(model, calibration_data)
    print(f"ppl_train: {ppl_train:.2f}")
    log_dict["ppl_train"] = ppl_train
    if args.log_wandb:
        wandb.log(log_dict)


if __name__ == "__main__":
    main()
