MODEL="Mistral-7B"
COMPR_PATH="/nfs/scistore19/alistgrp/osieberl/writeup/Optimal-Procrustes/gptq_models/fineweb_edu/Mistral-7B-v0.3/4" #"/path/to/your/data/folder"  # Modify this path to point to your local folder 
DEFAULT_LEVEL=4 # all layers quantized to 4 bit

if [[ $MODEL == Llama-2-7B ]]; then
    MODEL_ID=meta-llama/Llama-2-7b-hf
elif [[ $MODEL == Llama-3-8B ]]; then
    MODEL_ID=meta-llama/Meta-Llama-3-8B
elif [[ $MODEL == Meta-Llama-3.1-8B ]]; then
    MODEL_ID=meta-llama/Llama-3.1-8B
elif [[ $MODEL == Mistral-7B ]]; then
    MODEL_ID=mistralai/Mistral-7B-v0.3
elif [[ $MODEL == Llama-3.1-8B ]]; then
    MODEL_ID=meta-llama/Llama-3.1-8B
else
    echo "Unknown model"
    exit 1
fi

# for non-uniform compression pass the config path instead of default level (see lmeval.py for more details)
MODEL_LOADING_KWARGS=${MODEL_LOADING_KWARGS:-"--quant_weights_path $COMPR_PATH --quant_default_level $DEFAULT_LEVEL"} 

python lmeval.py \
    --model hf \
    --model_args pretrained=$MODEL_ID,low_cpu_mem_usage=True,dtype=float16 \
    $MODEL_LOADING_KWARGS \
    --tasks arc_easy,arc_challenge,winogrande,hellaswag,piqa \
    --batch_size 16

