'''
Code from https://github.com/blender-nlp/MolT5

```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''

import argparse
import csv
import os.path as osp
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import selfies as sf


def sf_encode(selfies):
    try:
        smiles = sf.decoder(selfies)
        return smiles
    except Exception:
        return None

def convert_to_canonical_smiles(smiles):
    if smiles is None:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
        return canonical_smiles
    else:
        return None
    
def build_evaluate_tuple(result:dict):
    # pred
    # func = lambda x: x.rsplit(']', 1)[0] + ']' if isinstance(x, str) else x
    func = lambda x: x
    result["pred_smi"] = convert_to_canonical_smiles(func(sf_encode(result["pred_self"])))
    # gt
    result["gt_smi"] = convert_to_canonical_smiles(sf_encode(result["gt_self"]))
    return result
    

def evaluate(input_file, morgan_r, verbose=False):
    outputs = []
    bad_mols = 0

    with open(osp.join(input_file)) as f:
        results = json.load(f)
        for i, result in enumerate(results):
            result = build_evaluate_tuple(result)
            try:
                gt_smi = result['gt_smi']
                ot_smi = result['pred_smi']
                
                gt_m = Chem.MolFromSmiles(gt_smi)
                ot_m = Chem.MolFromSmiles(ot_smi)

                if ot_m == None: raise ValueError('Bad SMILES')
                outputs.append((result['prompt'], gt_m, ot_m))
            except:
                bad_mols += 1
    validity_score = len(outputs)/(len(outputs)+bad_mols)
    if verbose:
        print('validity:', validity_score)


    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs

    for i, (desc, gt_m, ot_m) in enumerate(enum_list):

        if i % 100 == 0:
            if verbose: print(i, 'processed.')

        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    if verbose:
        print('Average MACCS Similarity:', maccs_sims_score)
        print('Average RDK Similarity:', rdk_sims_score)
        print('Average Morgan Similarity:', morgan_sims_score)
    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score


## TEST ##
def test_out_selfies_validity(args):
    with open(osp.join(args.input_file)) as f:
        results = json.load(f)
        bad_selfies = 0
        bad_mols = 0
        bad_gt_selfies = 0
        for i, result in enumerate(results):
            pred = result['pred_self']
            smi = sf_encode(pred)
            if not smi:
                bad_selfies += 1
            else:
                try:
                    m = Chem.MolFromSmiles(smi)
                    if m is None:
                        bad_mols += 1
                except:
                    bad_mols += 1
            gt = result['gt_self']
            gt_smi = sf_encode(gt)
            if not gt_smi:
                bad_gt_selfies += 1
        print('Pred: bad selfies:', bad_selfies)
        print('Pred: bad mols:', bad_mols)
        print('GT: bad selfies:', bad_gt_selfies)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='caption2smiles_example.json', help='path where test generations are saved')
    parser.add_argument('--morgan_r', type=int, default=2, help='morgan fingerprint radius')
    args = parser.parse_args()
    # test_out_selfies_validity(args)
    evaluate(args.input_file, args.morgan_r, True)
    
    
"""
# retrosynthesis
python -m llava.eval.molecule_metrics.fingerprint_metrics \
    --input_file=eval_result/moleculestm-retrosynthesis-5ep.jsonl 

# reagent_pred
python -m llava.eval.molecule_metrics.fingerprint_metrics \
    --input_file=eval_result/moleculestm-reagent_pred-5ep.jsonl 
    
# forward_pred
python -m llava.eval.molecule_metrics.fingerprint_metrics \
    --input_file=eval_result/moleculestm-forward_pred-5ep.jsonl 
"""