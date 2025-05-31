import math
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


from .clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .mpb import MemoryPromptBank
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='./'


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class TextPromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end'):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        if ctx_init:
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
       

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

    
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
       

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
       
            
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        

        if self.class_token_position == "end":
            
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=-2,
            )
        elif self.class_token_position == "middle":
            if self.split_idx is not None:
                half_n_ctx = self.split_idx 
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class MINT(nn.Module):
    def __init__(self, args, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end'):
        super(MINT, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
       
        self.text_prompt_learner = TextPromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position)
        self.criterion = criterion

        self.device = device
        self.args=args
        
        
        self.layer_e = [0] # layer that attach expert prompt : third to fifth layer by paper
        self.bank = MemoryPromptBank()
        self.topN = 2 
        self.bank_size=512
        self.prompt_num = 2
        self.bank.initBank(12, self.bank_size, self.prompt_num, 768, 768, self.device, embedding_layer=None)
        self.prompt_selected_train = []
        self.prompt_selected_sum_train = np.zeros((self.bank_size, ), dtype=int)
        self.prompt_selected_test = []
        self.prompt_selected_sum_test = np.zeros((self.bank_size, ), dtype=int)

        
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the text_prompt_learner (tunable prompt)
    def reset(self):
        self.text_prompt_learner.reset()
        

    def reset_classnames(self, classnames, arch):
        self.text_prompt_learner.reset_classnames(classnames, arch)

    def get_image_embedding(self, x):
        x = self.image_encoder.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + \
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        return x
    

    def similarity(self, q, k, topN):
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        sim = torch.matmul(q, k.T)  # (B, BankSize)
        
        dist = 1 - sim # Lower distance is better (higher sim)

        if self._training == True: # Or self.training if nn.Module's flag is used
            prompt_selected_sum = torch.Tensor(self.prompt_selected_sum_train).to(self.device)
            total = torch.sum(prompt_selected_sum)
            
            current_batch_size = q.shape[0] # Use actual batch size

          
            if total <= self.bank_size * current_batch_size: 
                if total > 0: 
                    freq = prompt_selected_sum / total
                    dist = dist * freq  
   

        _, idx = torch.topk(dist, topN, dim=1, largest=False) 
        
        dist_pick = []
        for b_idx in range(idx.shape[0]):
            pick = []
            for i_idx in range(idx.shape[1]):
                pick.append(dist[b_idx][idx[b_idx][i_idx]])
            dist_pick.append(torch.stack(pick))
        dist_selected = torch.stack(dist_pick)

        return dist_selected, idx
    
    
    def getPrompts(self, layer, bank, keys, distance): 
        B = keys.shape[0]
        
        if layer in self.layer_e: # Expert prompts
           
            num_prompts_per_group = bank.total 
            start_idx = layer * num_prompts_per_group
            end_idx = (layer + 1) * num_prompts_per_group
            prompts_for_current_group = bank.prompt_list[start_idx:end_idx]
            pTensor = torch.stack(list(prompts_for_current_group)).to(self.device, dtype=self.dtype)
            expert_prompts_choices = pTensor[keys] # Shape: (B, topN, self.prompt_num, D)
            fused_expert_prompts = torch.mean(expert_prompts_choices, dim=1) # Shape: (B, self.prompt_num, D)
    
            return fused_expert_prompts
        else:
            return None # No prompts for this layer
            

    def solve_selected_prompt(self, keys): # keys are (B, topN) tensor of indices
        keys_c = keys.cpu().numpy() # Convert to numpy array for iteration
        
        for b_keys in keys_c: # Iterate over batch
            for key_idx in b_keys: # Iterate over topN selections for this batch item
                if key_idx < len(self.prompt_selected_sum_train): # Boundary check
                    if self._training == True: # Or self.training
                        # self.prompt_selected_train.append(key_idx) # Appending all selected can be memory intensive
                        self.prompt_selected_sum_train[key_idx] += 1
                    else:
                        # self.prompt_selected_test.append(key_idx)
                        if key_idx < len(self.prompt_selected_sum_test): # Check for test sum array
                            self.prompt_selected_sum_test[key_idx] += 1
                # else: log warning or handle error for out-of-bounds key_idx
    
   
    def _forward_attn(self, attn_layer: nn.MultiheadAttention, x_norm, prompts_for_mha):
        prompts_for_mha = prompts_for_mha.expand(-1, x_norm.shape[1], -1)
        
       
        half_prompt_len = prompts_for_mha.shape[0] // 2
        prompts_k_insert = prompts_for_mha[:half_prompt_len, :, :]  # (P/2, B, C)
        prompts_v_insert = prompts_for_mha[half_prompt_len:, :, :] # (P/2, B, C)

        if x_norm.shape[0] == 0:
             raise ValueError("x_norm has zero sequence length in _forward_attn_with_prompts.")

        cls_token_norm = x_norm[0:1, :, :]      # (1, B, C)
        img_tokens_norm = x_norm[1:, :, :]      # (L-1, B, C)
        
       
        key_input_sequence = torch.cat([cls_token_norm, prompts_k_insert, img_tokens_norm], dim=0)
        value_input_sequence = torch.cat([cls_token_norm, prompts_v_insert, img_tokens_norm], dim=0)
        
        
        attn_output, _ = attn_layer(query=x_norm, key=key_input_sequence, value=value_input_sequence, need_weights=False)
     
        return attn_output

    def _forward_block(self, block, x, prompts_for_mha):
       
        x_normed_for_attn = block.ln_1(x) # Apply LayerNorm to x
        # block.attn is nn.MultiheadAttention
        attn_out = self._forward_attn(block.attn, x_normed_for_attn, prompts_for_mha)
        x = x + attn_out # Add to original x (residual connection for attention part)
        
        # MLP part
        x = x + block.mlp(block.ln_2(x)) # Add to x from attention (residual connection for MLP part)
        return x

    def _forward_image_transformer(self, x_embed, keys_from_similarity, distance_from_similarity):
        x = self.image_encoder.ln_pre(x_embed)
        x = x.permute(1, 0, 2)  # BNC -> LBC (L=N_tokens, B=batch_size, C=embed_dim)
        
        num_transformer_layers = len(self.image_encoder.transformer.resblocks)

        for i, blk in enumerate(self.image_encoder.transformer.resblocks):
            prompts_BPC = None # Shape: (Batch, PromptSeqLen, Dim)
            
            if i < num_transformer_layers: 
                if i in self.layer_e:
                    prompts_BPC = self.getPrompts(layer=i, bank=self.bank, keys=keys_from_similarity, distance=distance_from_similarity)
            
            if prompts_BPC is not None:
                prompts_PBC = prompts_BPC.permute(1, 0, 2)
                x = self._forward_block(blk, x, prompts_PBC)
            else:
                x = blk(x) # Standard block forward if no prompts for this layer
                
        x = x.permute(1, 0, 2)  # LBC -> BNC (back to Batch first)
        
        image_features = self.image_encoder.ln_post(x[:, 0, :]) # Output of CLS token
        
        if self.image_encoder.proj is not None: # Apply projection if it exists
            image_features = image_features @ self.image_encoder.proj
            
        return image_features # Final image features, shape (B, ProjDim)

    def get_text_features(self):
        text_features = []
        prompts = self.text_prompt_learner()
        tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)


    def forward(self, inputs, task_id=None):
        with torch.no_grad():
            x_embed = self.get_image_embedding(inputs) # Shape: (B, N_tokens, C_embed)

           
        with torch.no_grad(): 
            temp_x_embed = x_embed.clone().detach()
            temp_x = self.image_encoder.ln_pre(temp_x_embed)
            temp_x = temp_x.permute(1, 0, 2)  # BNC -> LBC

            all_cls_tokens_layers = []
            for blk in self.image_encoder.transformer.resblocks: 
                temp_x = blk(temp_x)
                cls_token_this_layer = temp_x[0]  # 形状: [B, Dim]
                all_cls_tokens_layers.append(cls_token_this_layer)

            stacked_cls_tokens = torch.stack(all_cls_tokens_layers, dim=0)


        num_keys_in_one_bank_group = self.bank.total 

        bank_layer_to_get = 0
        start_index = bank_layer_to_get * num_keys_in_one_bank_group
        end_index = (bank_layer_to_get + 1) * num_keys_in_one_bank_group

  
        keys_for_the_bank_layer = self.bank.key_list[start_index:end_index]
        
        kTensor1 = torch.stack(list(keys_for_the_bank_layer)).to(stacked_cls_tokens[3].device, dtype=self.dtype) # Shape: (BankSize, KeyDim)
        distance_selected1, keys_idx1 = self.similarity(stacked_cls_tokens[3], kTensor1, self.topN) 

        kTensor2 = torch.stack(list(keys_for_the_bank_layer)).to(stacked_cls_tokens[7].device, dtype=self.dtype) # Shape: (BankSize, KeyDim)
        distance_selected2, keys_idx2 = self.similarity(stacked_cls_tokens[7], kTensor2, self.topN)

        kTensor3 = torch.stack(list(keys_for_the_bank_layer)).to(stacked_cls_tokens[11].device, dtype=self.dtype) # Shape: (BankSize, KeyDim)
        distance_selected3, keys_idx3 = self.similarity(stacked_cls_tokens[11], kTensor3, self.topN)

        keys_idx = torch.cat((keys_idx1, keys_idx2, keys_idx3), dim=1)
        distance_selected = torch.cat((distance_selected1, distance_selected2, distance_selected3), dim=1)
        
        self.solve_selected_prompt(keys_idx) 
    
        image_features = self._forward_image_transformer(x_embed, keys_idx, distance_selected)
      
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

  
        text_features = self.get_text_features()
        
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits, distance_selected


def get_mint(clip_arch, test_set, args, device, n_ctx, ctx_init):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = MINT(args, device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init)

    return model

