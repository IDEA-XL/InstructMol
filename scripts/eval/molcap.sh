#!/bin/bash

GRAPH_TOWER="moleculestm"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/graphmvp.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/MoleculeSTM/molecule_model.pth"
else
    echo "Not supported graph tower"
fi

MODEL_PATH=checkpoints/llava-moleculestm-vicuna-v1-3-7b-finetune_lora
EPOCH=20
OUT_FILE=eval_result/$GRAPH_TOWER-chebi20-molcap-lora-${EPOCH}ep.jsonl

python -m llava.eval.model_molcap \
    --model-path $MODEL_PATH \
    --in-file assets/chebi-20_data/test.txt \
    --answers-file $OUT_FILE \
    --graph-checkpoint-path $INIT_CHECKPOINT_GNN \
    --model-base checkpoints/vicuna-v1-3-7b \
    --batch_size 4 \
    --debug 

# # evaluation 
# python -m llava.eval.eval_molcap \
#     --molcap_result_file $OUT_FILE \
#     --text2mol_bert_path checkpoints/scibert_scivocab_uncased