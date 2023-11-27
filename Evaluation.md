# Evaluation

## Property Prediction
### Classification task
```Shell
# Sampling
TASK="MoleculeNet"
GRAPH_TOWER=moleculestm
DATASET=bbbp # [bace, hiv]
EPOCH=20
python -m llava.eval.molecule_metrics.MoleculeNet_classification \
    --dataspace /cto_labs/AIDD/DATA/MoleculeNet \
    --dataset $DATASET \
    --model-path checkpoints/Graph-LLaVA/$TASK-llava-$GRAPH_TOWER-vicuna-v1-3-7b-finetune_lora \
    --graph-checkpoint-path checkpoints/MoleculeSTM/molecule_model.pth \
    --model-base checkpoints/vicuna-v1-3-7b \
    --batch_size 1 \
    --add-selfies \ # if set to True, then use InstructMol-GS to inference
    --debug 
# Evaluation
python -m llava.eval.molecule_metrics.property_metrics \
    --eval_result_file eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep.jsonl
```

### Regression task

Please download the regression test set from [Huggingface Mol-Instructions Dataset](https://huggingface.co/datasets/zjunlp/Mol-Instructions/blob/main/data/Molecule-oriented_Instructions.zip)
```Shell
# Sampling
TASK=property_pred
GRAPH_TOWER=moleculestm
EPOCH=5
python -m llava.eval.molecule_metrics.generate_sample \
    --task $TASK \
    --model-path LORA_MODEL_PATH \
    --in-file PATH_TO_PROPERTY_PREDICTION_TEST \
    --answers-file eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep.jsonl \
    --graph-checkpoint-path checkpoints/$GRAPH_TOWER/molecule_model.pth \
    --model-base checkpoints/vicuna-v1-3-7b \
    --batch_size 1 --temperature 0.2 --top_p 1.0 \
    --add-selfies \ # if set to True, then use InstructMol-GS to inference
    --debug 
# Evaluation
python -m llava.eval.molecule_metrics.property_metrics \
    --eval_result_file eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep.jsonl
```

## Molecule Description Generation
We use the [ChEBI-20 test dataset](assets/chebi-20_data/test.txt) for evaluation. 
```Shell
# Sampling
GRAPH_TOWER=moleculestm
EPOCH=20
OUT_FILE=eval_result/$GRAPH_TOWER-chebi20-molcap-lora-${EPOCH}ep.jsonl
python -m llava.eval.model_molcap \
    --model-path LORA_MODEL_PATH \
    --in-file assets/chebi-20_data/test.txt \
    --answers-file $OUT_FILE \
    --graph-checkpoint-path $INIT_CHECKPOINT_GNN \
    --model-base checkpoints/vicuna-v1-3-7b \
    --batch_size 1 \
    --add-selfies \ # if set to True, then use InstructMol-GS to inference
    --debug 
# Evaluation
python -m llava.eval.eval_molcap --molcap_result_file $OUT_FILE \
    --text2mol_bert_path checkpoints/scibert_scivocab_uncased
```


## Chemical Reaction Analysis
We take **Forward Reaction Prediction** as an example. For **Retrosynthesis Prediction** and **Reagent Prediction** task, just change the `$TASK` to `retrosynthesis` and `reagent_pred` respectively.


Please download the test set from [Huggingface Mol-Instructions Dataset](https://huggingface.co/datasets/zjunlp/Mol-Instructions/blob/main/data/Molecule-oriented_Instructions.zip)
```Shell
# Sampling
TASK=forward_pred
GRAPH_TOWER=moleculestm
EPOCH=5
python -m llava.eval.molecule_metrics.generate_sample \
    --task $TASK \
    --model-path LORA_MODEL_PATH \
    --in-file PATH_TO_FOWARD_REACTION_PREDICTION_TEST \
    --answers-file eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep.jsonl \
    --graph-checkpoint-path checkpoints/$GRAPH_TOWER/molecule_model.pth \
    --model-base checkpoints/vicuna-v1-3-7b \
    --batch_size 1 --temperature 0.2 --top_p 1.0 \
    --add-selfies \ # if set to True, then use InstructMol-GS to inference
    --debug 
# Evaluation
## Calculate the 'BLEU', 'exact match score', 'Levenshtein score' and 'validity'
python -m llava.eval.molecule_metrics.mol_translation_selfies \
    --input_file=eval_result/${GRAPH_TOWER}-${TASK}-${EPOCH}ep.jsonl
## Calculate the 'MACCS', 'RDK' and 'Morgan' similarity
python -m llava.eval.molecule_metrics.fingerprint_metrics \
    --input_file=eval_result/${GRAPH_TOWER}-${TASK}-${EPOCH}ep.jsonl
```