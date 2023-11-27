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

def mae_classify(path):
    from sklearn.metrics import mean_absolute_error
    LUMOs=[
        'Please provide the lowest unoccupied molecular orbital (LUMO) energy of this molecule.',
        'Please provide me with the LUMO energy value of this molecule.',
        'What is the LUMO energy of this molecule?',
        'I would like to know the LUMO energy of this molecule, could you please provide it?',
        'I would like to know the lowest unoccupied molecular orbital (LUMO) energy of this molecule, could you please provide it?',
        'Please provide the LUMO energy value for this molecule.',
        'I am interested in the LUMO energy of this molecule, could you tell me what it is?',
        'Could you give me the LUMO energy value of this molecule?',
        'Can you tell me the value of the LUMO energy for this molecule?',
        'What is the LUMO level of energy for this molecule?',
        'What is the lowest unoccupied molecular orbital (LUMO) energy of this molecule?',
        'Please provide the lowest unoccupied molecular orbital (LUMO) energy value for this molecule.',
    ]

    HOMOs=[
        'I would like to know the highest occupied molecular orbital (HOMO) energy of this molecule, could you please provide it?',
        'I am interested in the HOMO energy of this molecule, could you tell me what it is?',
        'Can you tell me the value of the HOMO energy for this molecule?',
        'I would like to know the HOMO energy of this molecule, could you please provide it?',
        'Please provide me with the HOMO energy value of this molecule.',
        'Please provide the highest occupied molecular orbital (HOMO) energy of this molecule.',
        'What is the HOMO level of energy for this molecule?',
        'What is the HOMO energy of this molecule?',
        'Could you give me the HOMO energy value of this molecule?',
        'Please provide the HOMO energy value for this molecule.',
        'What is the highest occupied molecular orbital (HOMO) energy of this molecule?',
        'Please provide the highest occupied molecular orbital (HOMO) energy value for this molecule.',
    ]

    HOMO_LUMOs=[
        'Please provide the energy separation between the highest occupied and lowest unoccupied molecular orbitals (HOMO-LUMO gap) of this molecule.',
        'Can you give me the energy difference between the HOMO and LUMO orbitals of this molecule?',
        'What is the energy separation between the HOMO and LUMO of this molecule?',
        'I need to know the HOMO-LUMO gap energy of this molecule, could you please provide it?',
        'What is the HOMO-LUMO gap of this molecule?',
        'Please provide the gap between HOMO and LUMO of this molecule.',
        'I would like to know the HOMO-LUMO gap of this molecule, can you provide it?',
        'Please give me the HOMO-LUMO gap energy for this molecule.',
        'Could you tell me the energy difference between HOMO and LUMO for this molecule?']
    
    LUMO_preds, LUMO_gts = [], []
    HOMO_preds, HOMO_gts = [], []
    GAP_preds, GAP_gts = [], []
    with open(path, 'rb') as f:
        list_data_dict = json.load(f)
        for entry in list_data_dict:
            prompt, gt, pred = entry["prompt"], entry["gt_self"], entry["pred_self"]
            prompt = prompt.split("The compound SELFIES sequence is:")[0].strip()
            if prompt in LUMOs:
                LUMO_preds.append(float(pred))
                LUMO_gts.append(float(gt))
            elif prompt in HOMOs:
                HOMO_preds.append(float(pred))
                HOMO_gts.append(float(gt))
            elif prompt in HOMO_LUMOs:
                GAP_preds.append(float(pred))
                GAP_gts.append(float(gt))
            else:
                raise ValueError(f"Cannot classified prompt", prompt)
    
    print("HOMO:", mean_absolute_error(HOMO_gts, HOMO_preds))
    print("LUMO:", mean_absolute_error(LUMO_gts, LUMO_preds))
    print("GAP:", mean_absolute_error(GAP_gts, GAP_preds))
        
    
    
if __name__ == '__main__':
    # test_reagent_pred_dataset()
    # test_forward_pred_dataset()
    # test_json_selfies()
    mae_classify("eval_result/moleculestm-property_pred-5ep-graph.jsonl")