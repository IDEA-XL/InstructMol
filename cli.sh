#!/bin/bash
MODEL_PATH="" # NOTE: Insert path to model here.(e.g., checkpoints/Graph-LLaVA/llava-moleculestm-vicuna-v1-3-7b-pretrain)

python -m llava.serve.cli_graph \
    --model-path $MODEL_PATH \
    --model-base checkpoints/vicuna-v1-3-7b \
    --graph-checkpoint-path checkpoints/graphmvp.pth \
    --debug
