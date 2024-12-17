export CUDA_VISIBLE_DEVICES=9

MODEL="mistralai/Mistral-7B-v0.3"
CALIB_DATA="fineweb_edu" 

SEQUENCE_LENGTH=8192
CALIB_TOKENS=131072

SPARSITY=0.375 # 12 removed blocks for Mistral-7B-v0.3 (32 blocks in total)

CONFIG_PATH="/path/to/your/data/folder"  # Modify this path to point to your local folder

GENERATIONS=$(awk "BEGIN {print int(($SPARSITY*32)*(32-($SPARSITY*32))/1.5)}") # k*(n-k)/1.5 (n: #blocks, k: #removed)

echo "Running with SPARSITY=$SPARSITY and GENERATIONS=$GENERATIONS"

# You might want to reduce eval_datasets or increase eval_every (does not impact the search, only the evaluation)
python evo_drop_search.py  \
    --model_name_or_path $MODEL \
    --sparsity $SPARSITY \
    --calibration_data $CALIB_DATA \
    --calibration_tokens $CALIB_TOKENS \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    --eval_every 5 \
    --eval_datasets fineweb_edu wikitext2 c4 \
    --eval_sequence_length $SEQUENCE_LENGTH \
    --population_size 1 \
    --generations $GENERATIONS \
    --offspring 32 \
    --initially_generated 64 \
    --initial_tokens 2048 \
    --survivors_per_selection 2 1 \
    --tokens_per_selection 2048 32768 \
    --fitness_fn kl \
    --use_fast_tokenizer \
    --drop_config_dir $CONFIG_PATH \
    --dtype float16 \
    --attn_implementation flash_attention_2 \
    --log_wandb

 

  
    