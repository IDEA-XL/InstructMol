# %%
import os 
import openai 
import json
import random

openai.api_key = "EMPTY"
openai.api_base = "http://192.168.81.82:8002/v1" # 当前部署在ctolab18上
model = "WEIGHTS/Llama-2-7b-chat-hf/"

def llm(prompt):
    completion = openai.ChatCompletion.create(
        model = model,
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 128,
        temperature = 0.5,
    )
    return completion.choices[0].message.content

# %%
import re
from tqdm import tqdm
from rdkit import Chem
from sklearn.metrics import mean_absolute_error


def is_valid_smiles(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return molecule is not None
    except:
        return False
    
def convert_to_standard_smiles(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            standard_smiles = Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True)
            return standard_smiles
        else:
            return None
    except:
        return None


def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def extract_floats_from_string(input_string):
    pattern = r"[-+]?\d*\.\d+|\d+"  # Regular expression pattern to match floats
    float_list = re.findall(pattern, input_string)
    return [float(num) for num in float_list]


class RegressionEvaluator(object):
    def __init__(self, metric=mean_absolute_error, verbose=False):
        self.metric = metric
        self.verbose = verbose
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
        if self.verbose:
            print(f"{self.metric.__name__} GT: {y_true}, Pred: {y_pred}, Diff: {abs(y_true - y_pred):.4f}")

    def summary(self):
        metric_value = self.metric(self.y_true, self.y_pred)
        return print(f"Final {self.metric.__name__}: {metric_value:.4f}")
    

# %%
from typing import List, Union


def template_build(instruction:str, selfies:List[str], gts:[float], template_prefix:str=None):
    template = f"""
        {template_prefix} 
    """
    for selfie, gt in zip(selfies, gts):
        template += f"""
        ###
        Instruction: "{instruction}"
        SELFIES: "{selfie}"
        Output: {gt}
        ###\n
        """
    return template
    
def prompt_build(instruction:str, selfies:str, template_selfies, template_gts, template_prefix:str):
    template = template_build(instruction, template_selfies, template_gts, template_prefix)
    prompt = f"""
        {template}
                
        Now, given the following molecule and based on the examples provided before, please provide the corresponding property value. If you cannot judge, just return a number you think is reasonable.
        
        SELFIES: "{selfies}"
        Output: 
    """
    return prompt

HOMO_TEMPLATE_PREFIX = """
    You're acting as a molecule property prediction assistant. Your task is to predict the given molecule descriptor using your experienced chemical knowledge. You'll be given SELFIES of molecule and you need to return a float result for the corresponding property result. \n
    Examples for HOMO energy prediction:\n
"""

LUMO_TEMPLATE_PREFIX = """
    You're acting as a molecule property prediction assistant. Your task is to predict the given molecule descriptor using your experienced chemical knowledge. You'll be given SELFIES of molecule and you need to return a float result for the corresponding property result. \n
    Examples for LUMO energy prediction:\n
"""

GAP_TEMPLATE_PREFIX = """
    You're acting as a molecule property prediction assistant. Your task is to predict the given molecule descriptor using your experienced chemical knowledge. You'll be given SELFIES of molecule and you need to return a float result for the corresponding property result. \n
    Examples for HOMO-LUMO energy gap prediction:\n
"""


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

attribute_map = {
    "LUMO": LUMOs,
    "HOMO": HOMOs,
    "GAP": HOMO_LUMOs,
}

prefix_map = {
    "LUMO": LUMO_TEMPLATE_PREFIX,
    "HOMO": HOMO_TEMPLATE_PREFIX,
    "GAP": GAP_TEMPLATE_PREFIX,
}

# evaluation setting
verbose=True
metrics_map = {
    "LUMO": RegressionEvaluator(verbose=verbose),
    "HOMO": RegressionEvaluator(verbose=verbose),
    "GAP": RegressionEvaluator(verbose=verbose),
}

example_data = read_json("/cto_labs/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/property_prediction_train.json")
FEW_SHOT = 5

def extract_fewshot(example_data, mode="HOMO"):
    results = []
    cursor = 0
    while cursor < len(example_data):
        if len(results) >= FEW_SHOT:
            break
        entry = example_data[cursor]
        instruction = entry["instruction"]
        if instruction not in attribute_map[mode]:
            cursor += 1
            continue
        results.append((entry["input"], entry["output"]))
        cursor += 1
    assert len(results) >= FEW_SHOT, f"not available to collect {FEW_SHOT} samples"
    return results

template_selfies_pools_HOMO = extract_fewshot(example_data, "HOMO")
template_selfies_pools_LUMO = extract_fewshot(example_data, "LUMO")
template_selfies_pools_GAP = extract_fewshot(example_data, "GAP")

template_selfies_map = {
    "LUMO": template_selfies_pools_LUMO,
    "HOMO": template_selfies_pools_HOMO,
    "GAP": template_selfies_pools_GAP,
}


# read test
test_data = read_json("/cto_labs/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/property_prediction_test.json")

for i, entry in enumerate(tqdm(test_data)):
    instruction = entry["instruction"]
    for mode,qs in attribute_map.items():
        if instruction in qs:
            break
    template_selfies = template_selfies_map[mode]
    prompt = prompt_build(
        instruction, entry["input"], [ts[0] for ts in template_selfies], 
        [ts[1] for ts in template_selfies], prefix_map[mode]
    )
    response = llm(prompt)
    try:
        pred = extract_floats_from_string(response)[0]
    except:
        print("error, randomly sample from FEW-shots")
        pred = random.choice([ts[1] for ts in template_selfies])
    metrics_map[mode].update(float(entry["output"]), float(pred))


# summary
print('Final Results:')
for mode, metric in metrics_map.items():
    print(mode, metric.summary())