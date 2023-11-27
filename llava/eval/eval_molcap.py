import logging
logger = logging.getLogger(__name__)

import argparse
import json
import os
from tqdm import tqdm
import numpy as np

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import torch
from transformers import BertTokenizerFast


def test_molcap_from_file(file, args):
    tokenizer = BertTokenizerFast.from_pretrained(args.text2mol_bert_path)
    output_tokens = []
    gt_tokens = []
    meteor_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    with open(file, "r") as f:
        for i,log in tqdm(enumerate(json.load(f))):
            cid,pred,gt = log['cid'],log['text'],log['gt']
            output_tokens.append(tokenizer.tokenize(pred, truncation=True, max_length=512, padding='max_length'))
            output_tokens[i] = list(filter(('[PAD]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[CLS]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[SEP]').__ne__, output_tokens[i]))

            gt_tokens.append(tokenizer.tokenize(gt, truncation=True, max_length=512, padding='max_length'))
            gt_tokens[i] = list(filter(('[PAD]').__ne__, gt_tokens[i]))
            gt_tokens[i] = list(filter(('[CLS]').__ne__, gt_tokens[i]))
            gt_tokens[i] = [list(filter(('[SEP]').__ne__, gt_tokens[i]))]

            meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i]))
            rouge_scores.append(scorer.score(gt, pred))
    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    
    # extract top-10 meteor scores
    meteor_scores = np.array(meteor_scores)
    Start,K = 500,100
    idxes = np.argsort(meteor_scores)[::-1][Start:Start+K]
    cids = [log['cid'] for i,log in enumerate(json.load(open(file, "r"))) if i in idxes]
    cids.sort(key=lambda x: int(x))

    return {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": np.mean(meteor_scores),
        "ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]),
        "ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]),
        "ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]),
    }
    
    
def add_arguments(parser):
    parser.add_argument("--molcap_result_file", type=str, required=True)
    parser.add_argument("--text2mol_bert_path", type=str, default="checkpoints/scibert_scivocab_uncased")
    
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    result = test_molcap_from_file(args.molcap_result_file, args)
    print(result)
    
"""
python -m llava.eval.eval_molcap \
    --molcap_result_file eval_result/chebi20-molcap-lora-10ep.json \
    --text2mol_bert_path checkpoints/scibert_scivocab_uncased
"""
