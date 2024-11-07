MODEL="mistralai/Mistral-7B-v0.3"
SEQUENCE_LENGTH=8192
BIT_LEVEL=3

CALIB_DATA="fineweb_edu" 

CALIB_TOKENS=524288
EVAL_TOKENS=524288

COMPR_PATH="/path/to/your/data/folder"  # Modify this path to point to your local folder (run gptq.sh first to produce weights)

python evo_quant_search.py \
    --calibration_data  $CALIB_DATA \
    --model_name_or_path $MODEL \
    --calibration_tokens $CALIB_TOKENS \
    --offspring 128 \
    --eval_every 5 \
    --eval_datasets "fineweb_edu" "wikitext2" "c4" \
    --quant_weights_path $COMPR_PATH \
    --log_wandb \
    --eval_sequence_length $SEQUENCE_LENGTH \
    --eval_tokens $EVAL_TOKENS \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    --survivors_per_selection 16 4 1 \
    --tokens_per_selection 2048 16384 131072 \
    --generations 150 \
    --target_bitwidth $BIT_LEVEL \
    --dtype float16 \
    --attn_implementation flash_attention_2 \
    --fitness_fn kl \