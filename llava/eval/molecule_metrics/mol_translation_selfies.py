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

import pickle
import argparse
import csv
import json
import os.path as osp
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev
from rdkit import Chem
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


def evaluate(input_file, verbose=False):
    outputs = []

    with open(osp.join(input_file)) as f:
        results = json.load(f)
        for i, result in enumerate(results):
            result = build_evaluate_tuple(result)
            gt_self = result['gt_self']
            ot_self = result['pred_self']
            gt_smi = result['gt_smi']
            ot_smi = result['pred_smi']
            if ot_smi is None:
                continue
            outputs.append((result['prompt'], gt_self, ot_self, gt_smi, ot_smi))


    bleu_self_scores = []
    bleu_smi_scores = []

    references_self = []
    hypotheses_self = []
    
    references_smi = []
    hypotheses_smi = []

    for i, (des, gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):

        if i % 100 == 0:
            if verbose:
                print(i, 'processed.')

        gt_self_tokens = [c for c in gt_self]
        out_self_tokens = [c for c in ot_self]

        references_self.append([gt_self_tokens])
        hypotheses_self.append(out_self_tokens)
        
        if ot_smi is None:
            continue
        
        gt_smi_tokens = [c for c in gt_smi]
        ot_smi_tokens = [c for c in ot_smi]

        references_smi.append([gt_smi_tokens])
        hypotheses_smi.append(ot_smi_tokens)
        

    # BLEU score
    bleu_score_self = corpus_bleu(references_self, hypotheses_self)
    if verbose: print(f'SELFIES BLEU score', bleu_score_self)

    references_self = []
    hypotheses_self = []
    
    references_smi = []
    hypotheses_smi = []

    levs_self = []
    levs_smi = []

    num_exact = 0

    bad_mols = 0

    for i, (des, gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):

        hypotheses_self.append(ot_self)
        references_self.append(gt_self)

        hypotheses_smi.append(ot_smi)
        references_smi.append(gt_smi)
        
        try:
            m_out = Chem.MolFromSmiles(ot_smi)
            m_gt = Chem.MolFromSmiles(gt_smi)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
            #if gt == out: num_exact += 1 #old version that didn't standardize strings
        except:
            bad_mols += 1

        levs_self.append(lev(ot_self, gt_self))
        levs_smi.append(lev(ot_smi, gt_smi))


    # Exact matching score
    exact_match_score = num_exact/(i+1)
    if verbose:
        print('Exact Match:')
        print(exact_match_score)

    # Levenshtein score
    levenshtein_score_smi = np.mean(levs_smi)
    if verbose:
        print('SMILES Levenshtein:')
        print(levenshtein_score_smi)
        
    validity_score = 1 - bad_mols/len(outputs)
    if verbose:
        print('validity:', validity_score)
        
        
## TEST ##
def test_out_selfies_validity(args):
    with open(osp.join(args.input_file)) as f:
        results = json.load(f)
        bad_selfies = 0
        for i, result in enumerate(results):
            pred = result['pred_self']
            if not sf_encode(pred):
                print(i, pred, 'bad selfies')
                bad_selfies += 1
        print('bad selfies:', bad_selfies)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='caption2smiles_example.json', help='path where test generations are saved')
    args = parser.parse_args()
    # test_out_selfies_validity(args)
    evaluate(args.input_file, verbose=True)
    
    
"""
# retrosynthesis
python -m llava.eval.molecule_metrics.mol_translation_selfies \
    --input_file=eval_result/moleculestm-retrosynthesis-5ep.jsonl 

# reagent prediction
python -m llava.eval.molecule_metrics.mol_translation_selfies \
    --input_file=eval_result/moleculestm-reagent_pred-5ep.jsonl
    
# forward_pred
python -m llava.eval.molecule_metrics.mol_translation_selfies \
    --input_file=eval_result/moleculestm-forward_pred-5ep.jsonl 
"""