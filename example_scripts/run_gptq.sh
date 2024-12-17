MODEL="mistralai/Mistral-7B-v0.3"
SEQUENCE_LENGTH=8192
CALIB_DATA="fineweb_edu" 

NUM_TOKENS=8388608 # 1024*8192
BITS_LIST="2 3 4 5 6"
BITS_TO_LOAD=3 # Bitwidth loaded for sequential GPTQ
GROUP_SIZE=128

SAVE_DIR="/path/to/your/data/folder"  # Modify this path to point to your local folder (will be used by quant_search.sh)

# For Llama models, the pre_block_modules should be "model.embed_tokens model.rotary_emb"
torchrun --nnodes=1 --nproc-per-node=8 --master_port 29501 quant.py \
    \
    --model_name_or_path $MODEL \
    --quantizable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj)$' \
    --pre_block_modules model.embed_tokens \
    --block_modules model.layers \
    --post_block_modules model.norm lm_head \
    \
    --calibration_data $CALIB_DATA \
    --calibration_tokens $NUM_TOKENS \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    \
    --bitwidth_options $BITS_LIST \
    --calibration_bitwidth $BITS_TO_LOAD \
    --group_size $GROUP_SIZE \
    --perchannel \
    \
    --low_cpu_mem_usage \
    --cpu_offload_modules \
    --cpu_offload_activations \
    --verbose \
    \
    --dtype float16 \
    --attn_implementation flash_attention_2 \
    \
    --save_dir $SAVE_DIR \
    --log_wandb