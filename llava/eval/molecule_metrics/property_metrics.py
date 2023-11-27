import json
import argparse
from sklearn.metrics import mean_absolute_error
from typing import List

def compute_mae(eval_result_file:str, except_idxs:List[int]=[]):
    with open(eval_result_file) as f:
        results = json.load(f)
        gts = []
        preds = []
        for i, result in enumerate(results):
            if i in except_idxs:
                continue
            pred = result['pred_self']
            gt = result['gt_self']
            gts.append(float(gt))
            preds.append(float(pred))
        return mean_absolute_error(gts, preds)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_result_file", type=str, required=True)
    args = parser.parse_args()
    # read except_idxs
    with open('/cto_labs/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/property_overlap.txt', 'r') as f:
        except_idxs = [int(line.split('\t')[0]) for line in f.readlines()]
    mae = compute_mae(args.eval_result_file, except_idxs)
    print(mae)
    
    
"""
# property_pred
TASK=property_pred
EPOCH=5
GRAPH_TOWER=moleculestm
python -m llava.eval.molecule_metrics.property_metrics \
    --eval_result_file=eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep.jsonl
"""