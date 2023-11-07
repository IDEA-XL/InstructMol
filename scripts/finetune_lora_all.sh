#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

GRAPH_TOWER="moleculestm"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/graphmvp.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/MoleculeSTM/molecule_model.pth"
else
    echo "Not supported graph tower"
fi

CHECKPOINT_FOLDER_PREFIX="./checkpoints/Graph-LLaVA"
TASK="all"

# 8 GPUs

deepspeed llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path "" \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter $CHECKPOINT_FOLDER_PREFIX/llava-$GRAPH_TOWER-$MODEL_VERSION-pretrain/checkpoint-48000/mm_projector.bin \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/graph-text-molgen/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 16 \
    --report_to none
