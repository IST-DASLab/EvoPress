# EvoPress

Code for paper [EvoPress: Towards Optimal Dynamic Model Compression via Evolutionary Search](https://arxiv.org/abs/2410.14649).
 
## Usage

### Repository structure
---

- ```scripts/``` —  contains bash scripts with the required arguments to run the method
- ```src/``` —  directory for helper methods and utility functions 
- ```evo_drop_search.py``` — evolutionary depth pruning 
- ```drop_scoring.py``` — scoring based baseline methods for depth pruning 
- ```brute_force_drop.py``` — brute force depth pruning
- ```evo_prune_search.py``` — evolutionary unstructured sparsity allocation
- ```prune.py``` — SparseGPT unstructured pruning (preparation of database for EvoPress) 
- ```owl_prune.py``` — SparseGPT unstructured pruning (preparation of database for OWL)
- ```evo_quant_search.py``` — evolutionary quantization bitwidth allocation
- ```quant.py``` — GPTQ quantization (preparation of database for EvoPress) 
- ```compute_layer_errors.py``` — compute NMSE for Dynamic Programming (DP) solver 
- ```dp_search.py``` — script to run DP solver on top of configuration produced by  `compute_layer_errors.py` 
- ```lmeval.py``` — LM Eval Harness evalution script 
- ```eval_ppl.py``` — perplexity evalution script

### Calibration data
---

We provide 3 options for calibration data: `wikitext2`, `c4`, `fineweb_edu`.
We recommend using the latter one for the best results. In our experiments we used **8M** tokens
for calibration. To prepare a specific amount of calibration data specify
`--calibration_tokens`. By default we trim the calibration sequence length to the maximal context length.
However, for some models, context length may be too long to fit into memory. We 
set `--calibration_sequence_length` to `8k` for models with context length `>=8k`.

In experiments we used `--calibration_tokens=2^23`and `--calibration_sequence_length=8192` for Llama-3-8B, Llama-3.1-8B, Phi-3-medium-128k-instruct, and `--calibration_sequence_length=4096` for Llama-2-7b.

### Multi-GPU
---

Some of the scripts (Unstructured Sparsity, Quantization) may operate in **distributed** mode
for faster execution. We recommend using `torchrun` to launch them:

```shell
torchrun --nnodes=1 --nproc-per-node=<NUM_GPU> <name_of_the_script.py> <args...>
```

### Depth pruning
---

We provide 3 versions for depth pruning:
* `evo_drop_search.py` — depth pruning via EvoPress
* `drop_scoring.py` — depth pruning via scoring methods
* `brute_force_drop.py` — depth pruning via brute force

To run EvoPress for depth pruning, execute `run_drop_search.sh` in the scripts folder.

### Unstructured Sparsity
---

We provide 2 version for unstructured pruning:
* `prune.py` —  SparseGPT unstructured pruning (preparation of database for EvoPress)
* `owl_prune.py` — SparseGPT unstructured pruning (preparation of database for OWL)

To run EvoPress for non-uniform unstructured pruning, first execute `run_sparse_gpt.sh` to generate the database and then `run_prune_search.sh` for the search.

### Quantization
---

We provide `quant.py` for producing the GPTQ database for EvoPress.

To run EvoPress for non-uniform quantization, first execute `run_gptq.sh` to generate the database and then `run_quant_search.sh` for the search.

### Evaluation
---

We provide `lmeval.py` and `eval_ppl.py` scripts for evaluation on [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) benchmarks and perplexity measurements. The interface of  `lmeval.py` mostly follows the instructions from the original. In addition, one should specify the path to sparse/quantized weights via `--sparse-weights-path`/`--quant-weights-path` argument and path to `.txt` with chosen compression levels via `--sparse-config-path`/`--quant-config-path` argument. We adopted `lm-eval==0.4.0` for evaluation. 

## Environment

This code was tested on the following environment:
```
pytorch                   2.4.0           py3.10_cuda12.1_cudnn9.1.0_0    pytorch
pytorch-cuda              12.1                 ha16c6d3_5    pytorch
cuda                      12.1.0                        0    nvidia/label/cuda-12.1.0
transformers              4.43.4                   pypi_0    pypi
datasets                  2.21.0                   pypi_0    pypi
lm-eval                   0.4.0                    pypi_0    pypi
```

## Notes

Scripts `prune.py`, `owl_prune.py`, `quant.py` produce several versions of compressed representation
for each weight `(100-200 Gb)`. Make sure that you have sufficient amount of free space on drive before running. Additionally, when using KL-Divergence as the fitness function for the search, ensure you have enough RAM to store the logits, particularly for the models with 128K vocabulary size. Alternatively, we implemented TopK-KL-Divergence in `evo_quant_search.py`, which significantly reduces memory requirements. Preliminary experiments have shown this method to be comparably effective to KL-Divergence for $K \geq 512$.
