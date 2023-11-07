import argparse
import torch
import os
import json
from tqdm import tqdm
import random
import shortuuid
from typing import Generator

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, MM_ENCODER_CFG
from llava.mol_utils import check_smiles_validity
from llava.datasets.smiles2graph import smiles2graph

from typing import Dict
from transformers import TextStreamer
from torch_geometric.data import Data


MOLCAP_INSTRUCTIONS = [
    'Could you give me a brief overview of this molecule?',
    'Could you provide a description of this molecule?',
    'Describe this molecule.',
    'Please give me some details about this molecule.',
    'Provide a brief overview of this molecule.',
    'Provide a description of this molecule.',
    'What can you tell me about this molecule?'
]


def _convert_dict_to_Data(data_dict: Dict) -> Data:
    return Data(
        x=torch.asarray(data_dict['node_feat']),
        edge_attr=torch.asarray(data_dict['edge_feat']),
        edge_index=torch.asarray(data_dict['edge_index']),
    )
    

def iterate_test_files(
    args, 
    skip_first_line:bool=False,
    convert_smiles_to_graph:bool=False,
    batch_size:int=4,
)->Generator:
    with open(args.in_file, "rt") as f:
        if skip_first_line:
            f.readline()
        batch = []
        for i, line in enumerate(f.readlines()):
            line = line.rstrip("\n").split("\t")
            cid, smi, gt = line
            if convert_smiles_to_graph:
                graph = smiles2graph(smi)
                batch.append((cid, graph, gt))
            else:
                batch.append((cid, smi, gt))
            if len(batch) == batch_size:
                yield zip(*batch)
                batch = []
        if len(batch) > 0:
            yield zip(*batch)

def _length_test_file(args, skip_first_line:bool=False)->int:
    with open(args.in_file, "rt") as f:
        if skip_first_line:
            f.readline()
        return len(f.readlines())


def main(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # output file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    # graph encoder config
    mm_encoder_cfg = MM_ENCODER_CFG(init_checkpoint=args.graph_checkpoint_path)
    mm_encoder_cfg = mm_encoder_cfg.dict()
    
    tokenizer, model, _, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, mm_encoder_cfg=mm_encoder_cfg)

    # Sampling 
    batch_size = args.batch_size
    outs = []
    
    for cids, graphs, gts in tqdm(
        iterate_test_files(args, skip_first_line=True, convert_smiles_to_graph=True, batch_size=batch_size),
        total=_length_test_file(args, skip_first_line=True)//batch_size,
    ):  
        bs = len(cids)
        graph_tensors = [_convert_dict_to_Data(graph).to(device) for graph in graphs]
        cur_prompts = []
        input_ids_batch = []
        stopping_criteria_batch = []
        for _ in range(bs):
            cur_prompt = random.choice(MOLCAP_INSTRUCTIONS)
            qs = cur_prompt
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
            cur_prompts.append(cur_prompt)
            input_ids_batch.append(input_ids.squeeze(0))
            stopping_criteria_batch.append(stopping_criteria)
        # pad input_ids
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(
            input_ids_batch,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids_batch,
                graphs=graph_tensors,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=stopping_criteria_batch
            )

        outputs = []
        for i in range(bs):
            output = tokenizer.decode(output_ids[i, input_ids.shape[1]:]).strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()
            outputs.append(output)
        
        for cid, cur_prompt, gt, output in zip(cids, cur_prompts, gts, outputs):
            ans_id = shortuuid.uuid()
            outs.append(
                {"cid": cid,
                "prompt": cur_prompt,
                "text": output,
                "answer_id": ans_id,
                "model_id": model_name,
                "gt": gt,
                "metadata": {}}
            )

            if args.debug:
                print("\n", {"gt": gt, "outputs": output}, "\n")
    
    # for cid, graph, gt in tqdm(
    #     iterate_test_files(args, skip_first_line=True, convert_smiles_to_graph=True),
    #     total=_length_test_file(args, skip_first_line=True),
    # ):
    #     cur_prompt = random.choice(MOLCAP_INSTRUCTIONS)
    #     qs = cur_prompt
    #     if model.config.mm_use_im_start_end:
    #         qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
    #     conv = conv_templates[args.conv_mode].copy()
    #     conv.append_message(conv.roles[0], qs)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()
        
    #     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    #     graph_tensor = [_convert_dict_to_Data(graph).to(device)]
    #     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    #     keywords = [stop_str]
    #     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    #     with torch.inference_mode():
    #         output_ids = model.generate(
    #             input_ids,
    #             graphs=graph_tensor,
    #             do_sample=True,
    #             temperature=args.temperature,
    #             top_p=args.top_p,
    #             num_beams=args.num_beams,
    #             max_new_tokens=args.max_new_tokens,
    #             use_cache=True,
    #             stopping_criteria=[stopping_criteria]
    #         )

    #     outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    #     if outputs.endswith(stop_str):
    #         outputs = outputs[:-len(stop_str)]
    #     outputs = outputs.strip()
        
    #     ans_id = shortuuid.uuid()
    #     outs.append(
    #         {"cid": cid,
    #         "prompt": cur_prompt,
    #         "text": outputs,
    #         "answer_id": ans_id,
    #         "model_id": model_name,
    #         "gt": gt,
    #         "metadata": {}}
    #     )

    #     if args.debug:
    #         print("\n", {"gt": gt, "outputs": outputs}, "\n")
    
    json.dump(outs, ans_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--in-file", type=str, default="assets/chebi-20_data/test.txt")
    parser.add_argument("--answers-file", type=str, default="eval_result/answer.jsonl")
    parser.add_argument("--graph-checkpoint-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    main(args)


"""
python -m llava.eval.model_molcap \
    --model-path checkpoints/llava-vicuna-v1-3-7b-finetune_lora \
    --in-file assets/chebi-20_data/test.txt \
    --answers-file eval_result/chebi20-molcap-lora-10ep.jsonl \
    --graph-checkpoint-path checkpoints/graphmvp.pth \
    --model-base checkpoints/vicuna-v1-3-7b \
    --batch_size 1 \
    --debug 
"""