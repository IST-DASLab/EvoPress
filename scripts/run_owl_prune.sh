SPARSITY=${SPARSITY:-70}
MASTER_PORT=${MASTER_PORT:-29501}
NUM_GPUS=8

MODEL=Mistral-7B-v0.3
CALIBRATION_DATA=fineweb_edu
NUM_TOKENS=8388608 # 1024*8192

SAVE_DIR="/path/to/your/data/folder"  # Modify this path to point to your local folder 

if [[ $MODEL == Llama-2-7b-hf ]]; then
    MODEL_ID=meta-llama/Llama-2-7b-hf
    SEQUENCE_LENGTH=4096
    PRE_BLOCK_MODULES="model.embed_tokens model.rotary_emb"
elif [[ $MODEL == Meta-Llama-3-8B ]]; then
    MODEL_ID=meta-llama/Meta-Llama-3-8B
    SEQUENCE_LENGTH=8192
    PRE_BLOCK_MODULES="model.embed_tokens model.rotary_emb"
elif [[ $MODEL == Meta-Llama-3.1-8B ]]; then
    MODEL_ID=meta-llama/Llama-3.1-8B
    SEQUENCE_LENGTH=8192
    PRE_BLOCK_MODULES="model.embed_tokens model.rotary_emb"
elif [[ $MODEL == Mistral-7B-v0.3 ]]; then
    MODEL_ID=mistralai/Mistral-7B-v0.3
    SEQUENCE_LENGTH=8192
    PRE_BLOCK_MODULES="model.embed_tokens"
elif [[ $MODEL == Phi-3-medium-128k-instruct ]]; then
    MODEL_ID=microsoft/Phi-3-medium-128k-instruct
    SEQUENCE_LENGTH=8192
    PRE_BLOCK_MODULES="model.embed_tokens"
else
    echo "Unknown model"
    exit 1
fi


torchrun --nnodes=1 --nproc-per-node=${NUM_GPUS} --master_port=${MASTER_PORT} owl_prune.py \
    \
    --model_name_or_path $MODEL_ID \
    --prunable_modules '.*layers.*((q|k|v|o|down|up|gate)_proj)$' \
    --pre_block_modules $PRE_BLOCK_MODULES \
    --block_modules model.layers  \
    \
    --calibration_data ${CALIBRATION_DATA} \
    --calibration_tokens ${NUM_TOKENS} \
    --calibration_sequence_length ${SEQUENCE_LENGTH} \
    \
    --sparsity "0.${SPARSITY}" \
    --rel_damp 0.01 \
    --owl_lambda 0.02 0.05 0.08 0.1 0.2 \
    --owl_m 3 5 7 10 \
    \
    --cpu_offload_modules \
    --cpu_offload_activations \
    --verbose \
    \
    --save_dir ${SAVE_DIR}
