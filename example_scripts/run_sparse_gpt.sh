MODEL="mistralai/Mistral-7B-v0.3"
SEQUENCE_LENGTH=8192
SPARSITY=0.7

CALIB_DATA="fineweb_edu"  
NUM_TOKENS=8388608 # 1024*8192
WEIGHTS_DIFF=1000000 
NUM_LEVELS=8

SAVE_DIR="/path/to/your/data/folder"  # Modify this path to point to your local folder (will be used by prune_search.sh)

# For Llama models, the pre_block_modules should be "model.embed_tokens model.rotary_emb"
torchrun --nnodes=1 --nproc-per-node=4 --master_port 29501 prune.py \
    \
    --model_name_or_path $MODEL \
    --prunable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj)$' \
    --pre_block_modules model.embed_tokens \
    --block_modules model.layers \
    \
    --calibration_data $CALIB_DATA \
    --calibration_tokens $NUM_TOKENS \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    \
    --sparsity $SPARSITY \
    --weights_diff $WEIGHTS_DIFF \
    --num_levels $NUM_LEVELS \
    \
    --low_cpu_mem_usage \
    --cpu_offload_modules \
    --cpu_offload_activations \
    --verbose \
    \
    --attn_implementation flash_attention_2 \
    --dtype float16 \
    \
    --save_dir $SAVE_DIR