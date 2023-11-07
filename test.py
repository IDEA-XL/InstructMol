from dataclasses import dataclass, field
import json
from llava.datasets.reagent_pred_dataset import ReagentPredSupervisedGraphDataset
from llava.datasets.forward_pred_dataset import ForwardPredSupervisedGraphDataset
import transformers

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data.(.pkl)"})
    lazy_preprocess: bool = True
    is_multimodal: bool = False
    mm_use_im_start_end: bool = False


def test_reagent_pred_dataset():
    model_name_or_path = "/comp_robot/rentianhe/caohe/AIDD/LLaVA/checkpoints/vicuna-v1-3-7b"
    model_max_length = 2048
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )
    
    data_args = DataArguments(
        data_path="/shared_space/caohe/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/reagent_prediction.json",
        lazy_preprocess=True,
        is_multimodal=True
    )
    dataset = ReagentPredSupervisedGraphDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    print(dataset[0])
    print(len(dataset))
    
    
def test_forward_pred_dataset():
    model_name_or_path = "/comp_robot/rentianhe/caohe/AIDD/LLaVA/checkpoints/vicuna-v1-3-7b"
    model_max_length = 2048
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )
    
    data_args = DataArguments(
        data_path="/shared_space/caohe/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/forward_reaction_prediction.json",
        lazy_preprocess=True,
        is_multimodal=True
    )
    dataset = ForwardPredSupervisedGraphDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    print(dataset[0])
    print(len(dataset))
    

def test_json_selfies():
    import selfies
    data_path = "/shared_space/caohe/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/reagent_prediction.json"
    with open(data_path, "rb") as f:
        list_data_dict = json.load(f)
    for raw in list_data_dict:
        input_selfies, output_selfies = raw['input'], raw['output']
        try:
            input_smiles = selfies.decoder(input_selfies)
        except:
            print('cannot convert inp_selifes to smiles', input_selfies)
        try:
            output_smiles = selfies.decoder(output_selfies)
        except:
            print('cannot convert out_selifes to smiles', output_selfies)
        
        
    
    
if __name__ == '__main__':
    # test_reagent_pred_dataset()
    test_forward_pred_dataset()
    # test_json_selfies()