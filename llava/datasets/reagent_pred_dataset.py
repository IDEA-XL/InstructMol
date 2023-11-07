import os
import random
import json
import copy
import pickle
from typing import Dict, Optional, Sequence, List
import selfies
import torch
from torch.utils.data import Dataset
import transformers
from .preprocess import preprocess, preprocess_multimodal
from .smiles2graph import smiles2graph


def construct_instruct_question(product:str):
    """
    Construct instruct question for each graph
    """
    question_pools = [
        'Can you suggest some possible reagents that could have been used in the following chemical reaction?',
        'Give some possible reagents that could have been used in the following chemical reaction.',
        'Please propose potential reagents that might have been utilized in the provided chemical reaction.',
        'Please provide possible reagents based on the following chemical reaction.',
    ]
    question = random.choice(question_pools)
    question += f"\nThe product is {product}"
    return question


class ReagentPredSupervisedGraphDataset(Dataset):
    """We use MolInstruction https://huggingface.co/datasets/zjunlp/Mol-Instructions/viewer/Molecule-oriented%20Instructions/reagent_prediction (128K) """
    add_selfies = True
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                ):
        super(ReagentPredSupervisedGraphDataset, self).__init__()
        with open(data_path, "rb") as f:
            list_data_dict = json.load(f)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        
    def selfies2smiles(self, selfies_str):
        try:
            smiles_str = selfies.decoder(selfies_str)
        except:
            smiles_str = None
        return smiles_str

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        if not self.add_selfies:
            # insert product to the instruction end
            instruction = construct_instruct_question(product)
        else:
            instruction = raw['instruction'] + f" The reaction is {input}"
        if random.random() < 0.5:
            instruction = "<image>\n" + instruction
        else:
            instruction = instruction + "\n<image>"
        graph=smiles2graph(reactant_smiles)
        sources = dict(
            conversations=[
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        )
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        if graph is not None:
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(graph is not None))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # graph exist in the data
        if graph is not None:
            data_dict['graph'] = graph
        elif self.data_args.is_multimodal:
            raise ValueError("Graph does not exist in the data, but the model is multimodal")
        return data_dict