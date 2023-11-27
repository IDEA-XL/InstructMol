import os
import random
import copy
import pickle
from typing import Dict, Optional, Sequence, List
import torch
from torch.utils.data import Dataset
import transformers
import selfies
from .preprocess import preprocess, preprocess_multimodal

def smiles2selfies(smiles_str):
    try:
        selfies_str = selfies.encoder(smiles_str)
    except:
        selfies_str = None
    return selfies_str

class MoleculeNetSupervisedGraphDataset(Dataset):
    add_selfies = False
    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                ):
        super(MoleculeNetSupervisedGraphDataset, self).__init__()
        self.dataspace = data_path
        self.tokenizer = tokenizer
        self.list_data_dict = self._load_pickle()
        self.data_args = data_args
        if self.add_selfies:
            print("WARNING: Add SELFIES to the instruction")
        
    def _load_pickle(self):
        # load "bace" "bbbp" "hiv" three datasets
        split = "random" # "" for scaffold
        list_data_dict = []
        for dataset in ["bace", "bbbp", "hiv"]:
            with open(os.path.join(self.dataspace, dataset, "processed", f"instruct-{split}-train.pkl"), "rb") as f:
                list_data_dict += pickle.load(f)
        return list_data_dict
        
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
        if self.add_selfies:
            selfies_str = smiles2selfies(raw['SMILES'])
            instruction += f" The compound SELFIES sequence is: {selfies_str}"
        if random.random() < 0.5:
            instruction = "<image>\n" + instruction
        else:
            instruction = instruction + "\n<image>"
        graph, target = raw['graph'], str(raw['label'])
        sources = dict(
            conversations=[
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": target}
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