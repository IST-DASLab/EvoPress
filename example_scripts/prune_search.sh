MODEL="mistralai/Mistral-7B-v0.3"
SEQUENCE_LENGTH=8192

CALIB_DATA="fineweb_edu" 

CALIB_TOKENS=524288
EVAL_TOKENS=524288

COMPR_PATH="/path/to/your/data/folder"  # Modify this path to point to your local folder (run gptq.sh first to produce weights)

# You might want to reduce eval_datasets or increase eval_every (does not impact the search, only the evaluation)
python evo_prune_search.py \
    --model_name_or_path $MODEL \
    --sparse_weights_path $COMPR_PATH \
    --calibration_data  $CALIB_DATA \
    --calibration_tokens $CALIB_TOKENS \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    --eval_every 10 \
    --eval_datasets fineweb_edu wikitext2 c4 \
    --eval_tokens $EVAL_TOKENS \
    --eval_sequence_length $SEQUENCE_LENGTH \
    --generations 400 \
    --offspring 64 \
    --survivors_per_selection 8 2 1 \
    --tokens_per_selection 2048 16384 65536 \
    --fitness_fn kl \
    --dtype float16 \
    --attn_implementation flash_attention_2 \
    --log_wandb