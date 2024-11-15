import argparse
import random
import os
import copy
import numpy as np
from tqdm import trange
from typing import List, Optional

import torch
import torch.nn as nn
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
    layer_order_fn,
)
from src.metrics import compute_perplexity, compute_kl_div


def load_states(model, layers, layer_names, drop_state, quant_state, quant_weights_path):
    for i in range(len(layers)):
        block = layers[i]
        if drop_state[i]:
            make_dummy_forward(block, "attn+mlp")
        else:
            restore_forward(block)
            old_level = model.quant_state[i]
            new_level = quant_state[i]
            if new_level != old_level:
                for layer_name in layer_names:
                    layer = model.get_submodule(layer_name)
                    if layer_name.startswith(f"model.layers\.{i}\."):
                        layer.weight.data = torch.load(
                            os.path.join(quant_weights_path, layer_name, f"{new_level}.pth"),
                            map_location=layer.weight.device,
                        ).to(layer.weight.dtype)
    # Update model state
    model.quant_state = quant_state


def compute_fitness(model, data, fitness_fn, invert_fitness, target_logits: Optional[torch.Tensor] = None) -> float:
    sign = 1
    if invert_fitness:
        sign = -1

    if fitness_fn == "ppl":
        return sign * compute_perplexity(model, data)
    else:
        return sign * compute_kl_div(model, data, target_logits)


def selection(
    model: AutoModelForCausalLM,
    layers: List[nn.Module],
    layer_names: List[str],
    quant_weights_path,
    candidates,
    num_survive: int,
    calibration_data,
    num_tokens: int,
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
    for drop_state, quant_state in candidates:
        load_states(model, layers, layer_names, drop_state, quant_state, quant_weights_path)
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
    # Quantization params
    parser.add_argument(
        "--target_bitwidth",
        type=float,
        required=True,
        help="Base level for all layers. If no integer, initialize random with this average",
    )
    parser.add_argument(
        "--bitwidth_options",
        nargs="+",
        type=int,
        required=True,
        help="List of bitwidths to quantize the model.",
    )
    parser.add_argument("--quant_weights_path", type=str, required=True, help="Path to quantized weights")
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size between adjacent levels",
    )
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

    layers = get_layers(model)
    blocks_to_remove = int(args.sparsity * len(layers))
    print(f"Removing {blocks_to_remove} blocks")
    total_blocks = len(layers)

    for layer in layers:
        dummy_initialize(layer)

    # Prepare layers and initial state
    layer_names = []
    for layer_name in os.listdir(args.quant_weights_path):
        if os.path.isdir(os.path.join(args.quant_weights_path, layer_name)):
            layer_names.append(layer_name)
    # Sort layers
    layer_names = sorted(layer_names, key=layer_order_fn)

    initial_population_candidates = (
        []
    )  # store initially generated search points (only take fittest for first population)

    while len(initial_population_candidates) < args.initially_generated:
        drop_state = [False] * total_blocks
        quant_state = [int(args.target_bitwidth)] * total_blocks
        remove_ind = random.sample(range(total_blocks), blocks_to_remove)
        for ind in remove_ind:
            drop_state[ind] = True
        if (drop_state, quant_state) in initial_population_candidates:  # avoid duplicates
            continue
        initial_population_candidates.append((drop_state, quant_state))

    model.quant_state = quant_state

    population, train_fitnesses = selection(
        model=model,
        layers=layers,
        layer_names=layer_names,
        quant_weights_path=args.quant_weights_path,
        candidates=initial_population_candidates,
        num_survive=args.population_size,
        calibration_data=calibration_data,
        invert_fitness=args.invert_fitness,
        num_tokens=args.initial_tokens,
        fitness_fn=args.fitness_fn,
    )

    log_dict = {}

    for gen_id in range(args.generations):
        print(f"Generation {gen_id + 1}/{args.generations}")
        print(f"Train fitness {train_fitnesses[0]:.2e}")

        for k, (drop_state, quant_state) in enumerate(population):
            print(f"Sample {k}")
            print(f"drop_state: {drop_state}")
            print(f"quant_state: {quant_state}")

        load_states(model, layers, layer_names, *population[0], args.quant_weights_path)
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

        ### Layer dropping mutation ###

        offspring_list = []
        # Generate offspring by Mutation
        while len(offspring_list) < args.offspring:
            offspring = copy.deepcopy(random.choice(population))

            # Mutation
            num_flips = min(
                random.randint(1, args.max_mutations), random.randint(1, args.max_mutations)
            )  # bias towards lower values
            for _ in range(num_flips):

                remove_ind = random.randint(0, total_blocks - 1)
                while offspring[0][remove_ind]:
                    remove_ind = random.randint(0, total_blocks - 1)
                # Get bitwidth of removed layer
                remove_bitwidth = offspring[1][remove_ind]

                add_ind = random.randint(0, total_blocks - 1)
                while not offspring[0][add_ind]:
                    add_ind = random.randint(0, total_blocks - 1)

                offspring[0][remove_ind] = True
                offspring[0][add_ind] = False
                offspring[1][add_ind] = remove_bitwidth

            if offspring in offspring_list or offspring in population:  # avoid duplicates
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
                layer_names=layer_names,
                quant_weights_path=args.quant_weights_path,
                candidates=offspring_list,
                num_survive=args.population_size,
                calibration_data=calibration_data,
                invert_fitness=args.invert_fitness,
                num_tokens=num_tokens,
                fitness_fn=args.fitness_fn,
            )

        population = offspring_list

        ### Quantization mutation ###

        offspring_list = []
        # Generate offspring by Mutation
        while len(offspring_list) < args.offspring:
            offspring = copy.deepcopy(random.choice(population))

            # Mutation
            num_flips = min(
                random.randint(1, args.max_mutations), random.randint(1, args.max_mutations)
            )  # bias towards lower values
            for _ in range(num_flips):

                decr_ids = [
                    i
                    for i in range(total_blocks)
                    if not offspring[0][i] and offspring[1][i] > min(args.bitwidth_options)
                ]
                if len(decr_ids) == 0:
                    continue
                decr_id = random.choice(decr_ids)

                incr_ids = [
                    i
                    for i in range(total_blocks)
                    if not offspring[0][i] and offspring[1][i] < max(args.bitwidth_options)
                ]
                if len(incr_ids) == 0:
                    continue
                incr_id = random.choice(incr_ids)

                offspring[1][decr_id] -= args.step_size
                offspring[1][incr_id] += args.step_size

            if offspring in offspring_list or offspring in population:  # avoid duplicates
                continue

            offspring_list.append(offspring)

        if len(offspring_list) > 0:
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
                    layer_names=layer_names,
                    quant_weights_path=args.quant_weights_path,
                    candidates=offspring_list,
                    num_survive=args.population_size,
                    calibration_data=calibration_data,
                    invert_fitness=args.invert_fitness,
                    num_tokens=num_tokens,
                    fitness_fn=args.fitness_fn,
                )

            population = offspring_list
        else:
            print(f"No bitwidth mutation options")

    # Final config
    final_drop_state, final_quant_state = population[0]

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        # Save layer drop config
        with open(os.path.join(args.save_dir, "layer_drop_and_quant_config.txt"), "w") as f:
            for i, (is_dropped, b) in enumerate(zip(final_drop_state, final_quant_state)):
                f.write(f"model.blocks.{i}: ({is_dropped}, {b})\n")

    print("Final configuration:")
    for i, (is_dropped, b) in enumerate(zip(final_drop_state, final_quant_state)):
        print(f"model.blocks.{i}: ({is_dropped}, {b})\n")

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
