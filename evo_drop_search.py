import argparse
import random
import os
import copy
import numpy as np
from tqdm import trange
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.data_utils import get_data
from src.common_utils import fix_seed
from src.model_utils import (
    get_layers,
    get_attn_layer_name,
    get_mlp_layer_name,
    make_dummy_forward,
    dummy_initialize,
    restore_forward,
)
from src.metrics import compute_perplexity, compute_kl_div


def get_layer_drop_config(removed_state) -> List[str]:
    num_blocks = len(removed_state["attn"])
    drop_config = ["none"] * num_blocks
    for i in range(num_blocks):
        if removed_state["attn"][i] and removed_state["mlp"][i]:
            drop_config[i] = "attn+mlp"
        elif removed_state["attn"][i]:
            drop_config[i] = "attn"
        elif removed_state["mlp"][i]:
            drop_config[i] = "mlp"
    return drop_config


def get_legal_mask(legal_to_drop_path, num_blocks):
    if legal_to_drop_path is None:
        legal_to_drop = {"attn": [True] * num_blocks, "mlp": [True] * num_blocks}
        return legal_to_drop

    with open(legal_to_drop_path, "r") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]

    assert (
        len(lines) == num_blocks
    ), "Number of blocks in model and legal_to_drop file do not match (If two_consecutive is set, number of blocks should be half of the model)"

    legal_to_drop = {"attn": [False] * len(lines), "mlp": [False] * len(lines)}
    for i in range(len(lines)):
        if lines[i] == "attn+mlp":
            legal_to_drop["attn"][i] = True
            legal_to_drop["mlp"][i] = True
        elif lines[i] == "attn":
            legal_to_drop["attn"][i] = True
        elif lines[i] == "mlp":
            legal_to_drop["mlp"][i] = True
    return legal_to_drop


# check if only blocks are dropped that are allowed to be dropped
def is_valid_state(removed_state, legal_to_drop):
    for subblock_type in ["attn", "mlp"]:
        for i in range(len(legal_to_drop[subblock_type])):
            if not legal_to_drop[subblock_type][i] and removed_state[subblock_type][i]:
                return False
    return True


def load_states(model, layers, removed_state, drop_two_consecutive):
    removed_state = copy.deepcopy(removed_state)
    if drop_two_consecutive:  # decompress: duplicate every entry
        removed_state["attn"] = [removed_state["attn"][i // 2] for i in range(2 * len(removed_state["attn"]))]
        removed_state["mlp"] = [removed_state["mlp"][i // 2] for i in range(2 * len(removed_state["mlp"]))]

    for subblock_type in ["attn", "mlp"]:
        for j in range(len(removed_state[subblock_type])):
            if subblock_type == "attn":
                subblock = getattr(layers[j], get_attn_layer_name(model))
            else:
                subblock = getattr(layers[j], get_mlp_layer_name(model))
            if removed_state[subblock_type][j]:
                make_dummy_forward(subblock, subblock_type)
            else:
                restore_forward(subblock)


def compute_fitness(model, data, fitness_fn, invert_fitness, target_logits: Optional[torch.Tensor] = None) -> float:
    sign = 1
    if invert_fitness:
        sign = -1

    if fitness_fn == "ppl":
        return sign * compute_perplexity(model, data)
    else:
        return sign * compute_kl_div(model, data, target_logits)


def selection(
    model,
    layers,
    candidates,
    num_survive: int,
    calibration_data,
    num_tokens: int,
    drop_two_consecutive: bool,
    invert_fitness: bool,
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
        load_states(model, layers, candidate, drop_two_consecutive)
        fitness = compute_fitness(model, calibration_minibatch, fitness_fn, invert_fitness, target_logits_minibatch)
        fitnesses.append(fitness)
    # Keep only best
    best_ids = np.argsort(fitnesses)[:num_survive]
    return [candidates[i] for i in best_ids], [fitnesses[i] for i in best_ids]


def parse_args():
    parser = argparse.ArgumentParser()
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
    # Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", type=int, required=True, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", type=int, required=True, help="Length of calibration sequences."
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["fineweb_edu", "wikitext2", "c4"],
        help="Datasets used for evaluation",
    )
    parser.add_argument("--no_eval", action="store_true", help="Whether to skip evaluation")
    parser.add_argument("--eval_every", default=1, type=int, help="Eval every # generations.")
    parser.add_argument("--eval_tokens", default=524288, type=int, help="Number of tokens for evaluation.")
    parser.add_argument("--eval_sequence_length", default=None, type=int, help="Length of evaluation sequences.")
    # Sparsification params
    parser.add_argument("--sparsity", type=float, required=True, help="Fraction of layers to drop.")
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")
    # Evolutionary Search params
    parser.add_argument("--fitness_fn", choices=["ppl", "kl"], default="kl", help="Fitness function.")
    parser.add_argument("--generations", required=True, type=int, help="Number of generations in evolutionary search")
    parser.add_argument("--offspring", type=int, required=True, help="Number of offspring generated in each generation")
    parser.add_argument("--population_size", type=int, default=1, help="Population size in evolutionary search")
    parser.add_argument(
        "--initially_generated",
        type=int,
        required=True,
        help="Number of search points generated in the beginning; fittest are selected for the initial population",
    )
    parser.add_argument(
        "--initial_tokens",
        type=int,
        required=True,
        help="Number of calibration tokens used for the initial generation",
    )
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
    # Evolutionary Search ablation params
    parser.add_argument(
        "--invert_fitness", action="store_true", help="Whether to invert the fitness function (search for worst)"
    )
    parser.add_argument("--max_mutations", type=int, default=3, help="Maximum number of mutations in offspring")
    parser.add_argument(
        "--legal_to_drop_path",
        type=str,
        default=None,
        help="Path to legal_to_drop file. A block can only be dropped if it is dropped in legal_to_drop configuration.",
    )
    parser.add_argument("--drop_entire_block", action="store_true", help="Whether to drop entire block (attn+mlp).")
    parser.add_argument(
        "--drop_two_consecutive",
        action="store_true",
        help="Only drop pairs of consecutive blocks (first and second, third and fourth,...). Can only be set when entire blocks are dropped.",
    )
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    # Save params
    parser.add_argument("--save_dir", type=str, help="Where to save sparse model.")
    parser.add_argument("--drop_config_dir", type=str, help="Where to save layer drop config.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Sanity checks
    assert len(args.survivors_per_selection) == len(
        args.tokens_per_selection
    ), "Lists for selection survivors and tokens must have same length"
    assert args.survivors_per_selection[-1] == args.population_size, "Last stage should have population_size survivor"
    if args.drop_two_consecutive:
        assert args.drop_entire_block, "Can't drop two consecutive without dropping entire block"
        assert args.legal_to_drop_path == None, "Not implemented"
    # Get device and dtype
    assert torch.cuda.is_available()
    print(args.generations)
    device = f"cuda"
    dtype = getattr(torch, args.dtype)
    # Fix seed
    fix_seed(args.seed)
    # Init W&B logger
    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    print(model.config.model_type)
    print(model)
    model.config.use_cache = False  # do not use cache
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast=args.use_fast_tokenizer
    )
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or model.config.max_position_embeddings
    calibration_data = get_data(
        args.calibration_data,
        args.calibration_tokens,
        args.calibration_sequence_length,
        tokenizer,
        train=True,
    )
    # Load evaluation data
    args.sequence_length = args.eval_sequence_length or model.config.max_position_embeddings
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

    layers = get_layers(model)
    blocks_to_remove = int(args.sparsity * len(layers))
    print(f"Removing {blocks_to_remove} blocks")
    total_blocks = len(layers)

    if args.drop_two_consecutive:
        assert total_blocks % 2 == 0 and blocks_to_remove % 2 == 0, "Number of total and removed blocks must be even"
        total_blocks = total_blocks // 2  # view two consecutive blocks as one block
        blocks_to_remove = blocks_to_remove // 2

    for layer in layers:
        dummy_initialize(getattr(layer, get_attn_layer_name(model)))
        dummy_initialize(getattr(layer, get_mlp_layer_name(model)))

    legal_mask = get_legal_mask(
        args.legal_to_drop_path, total_blocks
    )  # mask of blocks that can be dropped (all blocks by default)

    initial_population_candidates = (
        []
    )  # store initially generated search points (only take fittest for first population)

    while len(initial_population_candidates) < args.initially_generated:
        removed_state = {"attn": [False] * total_blocks, "mlp": [False] * total_blocks}

        attn_legal_ind = [i for i in range(total_blocks) if legal_mask["attn"][i]]
        attn_remove_ind = random.sample(attn_legal_ind, blocks_to_remove)
        for ind in attn_remove_ind:
            removed_state["attn"][ind] = True

        mlp_legal_ind = [i for i in range(total_blocks) if legal_mask["mlp"][i]]
        mlp_remove_ind = random.sample(mlp_legal_ind, blocks_to_remove)
        for ind in mlp_remove_ind:
            removed_state["mlp"][ind] = True

        if args.drop_entire_block:
            removed_state["mlp"] = copy.deepcopy(removed_state["attn"])

        if removed_state in initial_population_candidates:  # avoid duplicates
            continue
        if not is_valid_state(removed_state, legal_mask):
            continue

        initial_population_candidates.append(removed_state)

    population, train_fitnesses = selection(
        model=model,
        layers=layers,
        candidates=initial_population_candidates,
        num_survive=args.population_size,
        calibration_data=calibration_data,
        invert_fitness=args.invert_fitness,
        drop_two_consecutive=args.drop_two_consecutive,
        num_tokens=args.initial_tokens,
        fitness_fn=args.fitness_fn,
        target_logits=target_logits,
    )

    log_dict = {}

    for gen_id in range(args.generations):
        print(f"Generation {gen_id + 1}/{args.generations}")
        print(f"Train fitness {train_fitnesses[0]:.2e}")

        for parent in population:
            print(f"Parent: attn: {[int(ele) for ele in parent['attn']]} mlp: {[int(ele) for ele in parent['mlp']]}")

        load_states(model, layers, population[0], args.drop_two_consecutive)
        log_dict["train_fitness"] = train_fitnesses[0]
        # Evaluate current search point
        if gen_id % args.eval_every == 0 and not args.no_eval:
            for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
                ppl_eval = compute_perplexity(model, eval_dataset)
                print(f"{eval_dataset_name}: {ppl_eval:.2f}")
                log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval

            full_train_ppl = compute_perplexity(model, calibration_data)
            print(f"full train ppl: {full_train_ppl:.2e}")
            log_dict["full_train_ppl"] = full_train_ppl

        if args.log_wandb:
            wandb.log(log_dict)

        offspring_list = []

        # Generate offspring by Mutation
        while len(offspring_list) < args.offspring:
            offspring = copy.deepcopy(random.choice(population))

            # Mutation
            num_flips = min(
                random.randint(1, args.max_mutations), random.randint(1, args.max_mutations)
            )  # bias towards lower values
            for _ in range(num_flips):
                remove_type = random.randint(0, 1)  # 0 remove attention, 1 remove mlp
                if remove_type == 0:
                    subblock_type = "attn"
                else:
                    subblock_type = "mlp"

                remove_ind = random.randint(0, total_blocks - 1)
                while offspring[subblock_type][remove_ind]:
                    remove_ind = random.randint(0, total_blocks - 1)

                add_ind = random.randint(0, total_blocks - 1)
                while not offspring[subblock_type][add_ind]:
                    add_ind = random.randint(0, total_blocks - 1)

                offspring[subblock_type][remove_ind] = True
                offspring[subblock_type][add_ind] = False

            if args.drop_entire_block:
                offspring["mlp"] = copy.deepcopy(offspring["attn"])

            if offspring in offspring_list or offspring in population:  # avoid duplicates
                continue

            if not is_valid_state(offspring, legal_mask):
                continue

            offspring_list.append(offspring)

        # Selection in multiple steps
        for num_survive, num_tokens in zip(args.survivors_per_selection, args.tokens_per_selection):
            if num_survive == args.survivors_per_selection[-1]:
                for i in range(
                    len(population)
                ):  # Elitist EA: Add search points in current generation to final selection step
                    if population[i] not in offspring_list:
                        offspring_list.append(population[i])

            offspring_list, train_fitnesses = selection(
                model=model,
                layers=layers,
                candidates=offspring_list,
                num_survive=num_survive,
                calibration_data=calibration_data,
                drop_two_consecutive=args.drop_two_consecutive,
                invert_fitness=args.invert_fitness,
                num_tokens=num_tokens,
                fitness_fn=args.fitness_fn,
                target_logits=target_logits,
            )

        population = offspring_list

        layer_drop_config = get_layer_drop_config(population[0])
        if args.drop_config_dir:
            os.makedirs(args.drop_config_dir, exist_ok=True)
            with open(os.path.join(args.drop_config_dir, "layer_drop_config.txt"), "w") as f:
                for line in layer_drop_config:
                    f.write(line + "\n")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        # Save model
        torch.save(model, os.path.join(args.save_dir, "final_model.pth"))
        # Save layer drop config
        with open(os.path.join(args.save_dir, "layer_drop_config.txt"), "w") as f:
            for line in layer_drop_config:
                f.write(line + "\n")

    print("Final configuration:")
    for line in layer_drop_config:
        print(line)

    # Final evaluation
    for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
        ppl_eval = compute_perplexity(model, eval_dataset)
        print(f"{eval_dataset_name}: {ppl_eval:.2f}")
        log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval

    full_train_ppl = compute_perplexity(model, calibration_data)
    print(f"full train ppl: {full_train_ppl:.2e}")
    log_dict["full_train_ppl"] = full_train_ppl
    if args.log_wandb:
        wandb.log(log_dict)


if __name__ == "__main__":
    main()
