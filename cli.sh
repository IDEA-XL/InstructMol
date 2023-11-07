#!/bin/bash
export HF_HOME=/comp_robot/rentianhe/caohe/cache
# python -m llava.serve.cli \
#     --model-path /comp_robot/rentianhe/caohe/AIDD/LLaVA/checkpoints/llava-v1-0719-336px-lora-vicuna-13b-v1.3 \
#     --model-base checkpoints/vicuna-v1-3-13b \
#     --image-file "https://llava-vl.github.io/static/images/view.jpg" 

python -m llava.serve.cli_graph \
    --model-path checkpoints/llava-vicuna-v1-3-7b-finetune_lora \
    --model-base checkpoints/vicuna-v1-3-7b \
    --graph-checkpoint-path checkpoints/graphmvp.pth \
    --debug
