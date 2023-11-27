#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_graph_tower
from .multimodal_projector.builder import build_xmodal_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_graph_tower"):
            self.graph_tower = build_graph_tower(config)
            self.mm_projector = build_xmodal_projector(config)

    def get_graph_tower(self):
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def initialize_graph_modules(self, model_args, fsdp=None):
        graph_tower = model_args.graph_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_graph_tower = graph_tower

        graph_tower = build_graph_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.graph_tower = [graph_tower]
        else:
            self.graph_tower = graph_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = graph_tower.hidden_size

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = build_xmodal_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_graph_tower(self):
        return self.get_model().get_graph_tower()

    def encode_mol(self, mol):
        # mol: graph data
        _, h_node = self.get_model().get_graph_tower().encode_mol(mol, proj=False, return_node_feats=True)
        # WARN: some weird thing happend if excute in bfloat16, so we force to cast to float32
        dtype = h_node.dtype
        if dtype == torch.bfloat16:
            h_node = h_node.float()
            graph_features = self.get_model().mm_projector.float()(h_node)
            graph_features = graph_features.to(dtype=dtype)
        else:
            graph_features = self.get_model().mm_projector(h_node)
        return graph_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, graphs
    ):
        graph_tower = self.get_graph_tower()
        if graph_tower is None or graphs is None or input_ids.shape[1] == 1:
            if past_key_values is not None and graph_tower is not None and graphs is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(graphs) is list:
            # skip `to_data_list`
            assert len(graphs) == input_ids.shape[0]
        else:
            graphs = graphs.to_data_list()
        graph_features = [self.encode_mol(graph) for graph in graphs]

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_graph_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (0. * self.get_model().mm_projector(graph_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_graph_idx += 1
                continue
            graph_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while graph_token_indices.numel() > 0:
                cur_graph_features = graph_features[cur_graph_idx]
                graph_token_start = graph_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[graph_token_start-1:graph_token_start]))
                    cur_new_input_embeds.append(cur_graph_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[graph_token_start+1:graph_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[graph_token_start:graph_token_start+1])
                        cur_labels = cur_labels[graph_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start]))
                    cur_new_input_embeds.append(cur_graph_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[graph_token_start+1:]
                cur_graph_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[graph_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[graph_token_start+1:]
                graph_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                if new_labels is not None:
                    new_attention_mask = []
                    _new_labels = new_labels
                    for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                        new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                        new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                        cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                        new_attention_mask.append(cur_new_attention_mask)
                    attention_mask = torch.stack(new_attention_mask, dim=0)
                    assert attention_mask.shape == new_labels.shape
                else:
                    attention_mask = torch.cat(
                        (torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device), attention_mask),
                        dim=1,
                    )
                    assert attention_mask.shape == new_input_embeds.shape[:2]
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_graph_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
