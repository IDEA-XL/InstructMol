# InstructMol: Multi-Modal Integration for Building a Versatile and Reliable Molecular Assistant in Drug Discovery


## Overview
<p align="center">
    <a> <img src="assets/static/teaser.png" width="100%"> </a>
</p>
The rapid evolution of artificial intelligence in drug discovery encounters challenges with generalization and extensive training, yet Large Language Models (LLMs) offer promise in reshaping interactions with complex molecular data. Our novel contribution, InstructMol, a multi-modal LLM, effectively aligns molecular structures with natural language via an instruction-tuning approach, utilizing a two-stage training strategy that adeptly combines limited domain-specific data with molecular and textual information. InstructMol showcases substantial performance improvements in drug discovery-related molecular tasks, surpassing leading LLMs and significantly reducing the gap with specialized models, thereby establishing a robust foundation for a versatile and dependable drug discovery assistant.

## Architecture
The diagram presented below provides an overview of the architectural design of the InstructMol model, along with its two-stage training paradigm. The example molecule in the figure is Terephthalaldehyde (CID 12173).
<p align="center">
    <a> <img src="assets/static/overview.png" width="80%"> </a>
</p>



## Contents
- [Install](#install)
- [Weights](#weights)
- [Dataset](#dataset)
- [CLI Inference](#cli-inference)
- [Train](#train)
- [Evaluation](#evaluation)

## Install
Mostly refer to LLaVA installation
1. Clone this repository and navigate to project folder

2. Install Package
- If you have any trouble install torch-geometric related packages, please refer to [guide-to-pyg-install](https://github.com/chao1224/GraphMVP#environments) for detailed instructions.
```Shell
conda create -n instructmol python=3.10 -y
conda activate instructmol
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# Install Graph related packages. We use torch-112 with CUDA-11.6, please change accordingly.
pip install -r requirements.txt
```

3. Install additional packages for training cases
```
pip install ninja
pip install flash-attn --no-build-isolation
```


## Weights

### Component Weights Download
Create a folder named `checkpoints` in the root directory of this project. 
```Shell
mkdir checkpoints
cd checkpoints
```
Download the following weights and put them in the `checkpoints` folder.
```Shell
# Under the checkpoints folder
# get the weights for the vicuna model (https://huggingface.co/lmsys/vicuna-7b-v1.3)
ln -s YOUR_PATH_TO_vicuna_v1_3_7b vicuna-v1-3-7b
# get the weights for MoleculeSTM model
mkdir MoleculeSTM
wget https://huggingface.co/chao1224/MoleculeSTM/resolve/main/demo/demo_checkpoints_Graph/molecule_model.pth -P MoleculeSTM
# download the weights for scibert_scivocab_uncased model (https://huggingface.co/allenai/scibert_scivocab_uncased)
ln -s YOUR_PATH_TO_scibert_scivocab_uncased scibert_scivocab_uncased
cd .. # back to the root directory
```
* [Optional] Get graphmvp weights, please refer to [GraphMVP weights download guidance](https://github.com/chao1224/GraphMVP#for-graphmvp-pre-training). 
    ```Shell
    mv YOUR_PATH_TO_graphmvp.pth checkpoints/
    ```

## CLI Inference
Chat with InstructMol without the need of Gradio interface. 
```Shell
#!/bin/bash
# NOTE: Insert path of model here.(e.g., checkpoints/Graph-LLaVA/llava-moleculestm-vicuna-v1-3-7b-pretrain)
MODEL_PATH="" 
python -m llava.serve.cli_graph \
    --model-path $MODEL_PATH \
    --model-base checkpoints/vicuna-v1-3-7b \
    --graph-checkpoint-path checkpoints/graphmvp.pth 
```


## Train
LLaVA training consists of two stages:

* **Stage 1: Alignment Pretraining.** Initial stage aligns molecules with text using a PubChem dataset of 330K pairs. Focuses on fine-tuning the alignment projector while keeping the graph encoder and LLM frozen to leverage pre-trained knowledge.
* **Stage 2: Task-specific Instruction Tuning.** Second stage targets compound property prediction, chemical reaction analysis, and molecule description generation. Utilizes task-specific instruction datasets and LoRA for LLM adaptation, retaining common-sense reasoning capabilities. Allows adaptable adaptors for specific needs or modular knowledge integration.

### Stage 1: Alignment Pretraining
See [pretrain.sh](scripts/pretrain.sh) for an example of how to run the pretraining stage.
- `$GRAPH_TOWER` can be chosen from `moleculestm` or `graphmvp`.

### Stage 2: Task-specific Instruction Tuning
You can train all specific tasks combine together [finetune_all.sh](scripts/all/finetune_lora_all.sh) or train them separately, (e.g., [molecule description generation task](scripts/finetune_lora_molcap.sh)).


## Evaluation
See [Evaluation.md](Evaluation.md) for detailed instructions on how to evaluate the model.
