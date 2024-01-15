import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from functools import reduce, partial

from models.vision_transformer import VisionTransformer


class PatchPrompter(nn.Module):
    def __init__(self, args):
        super(PatchPrompter, self).__init__()
        self.patch_size = args.patch_size
        self.prompt_size = args.prompt_size
        self.fg_size = self.patch_size - args.prompt_size * 2

        self.patch = nn.Parameter(torch.randn([1, 3, args.image_size, args.image_size]))

    def forward(self, x):
        _, _, h, w = x.size()

        fg_in_patch = torch.zeros([1, 3, self.fg_size, self.fg_size]).cuda()
        fg_in_patch = F.pad(fg_in_patch, (self.prompt_size, self.prompt_size, self.prompt_size, self.prompt_size), "constant", 1)
        mask = fg_in_patch.repeat(1, 1, h//self.patch_size, w//self.patch_size)
        self.prompt = self.patch * mask

        return x + self.prompt


class SharedPrompter(nn.Module):
    def __init__(self, args):
        super(SharedPrompter, self).__init__()
        self.patch_size = args.patch_size
        self.prompt_size = args.prompt_size
        self.fg_size = self.patch_size - args.prompt_size * 2

        self.patch = nn.Parameter(torch.randn([1, 3, self.patch_size, self.patch_size]))

    def forward(self, x):
        _, _, h, w = x.size()

        fg_in_patch = torch.zeros([1, 3, self.fg_size, self.fg_size]).cuda()
        fg_in_patch = F.pad(fg_in_patch, (self.prompt_size, self.prompt_size, self.prompt_size, self.prompt_size), "constant", 1)
        
        mask = fg_in_patch.repeat(1, 1, h//self.patch_size, w//self.patch_size)
        patch = self.patch.repeat(1, 1, h//self.patch_size, w//self.patch_size)

        self.prompt = patch * mask

        return x + self.prompt
        

class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


def build_ViTwSPT(backbone, args):
    model = ViTwSPT(args)

    model.load_state_dict(backbone.state_dict(), False)
    model.freeze()

    return model


class ViTwSPT(VisionTransformer):
    def __init__(self, args):
        super().__init__(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.depth = 12
        self.num_patches = args.image_size*args.image_size // (16*16)
        self.prompters = [nn.Parameter(torch.randn([1, 1, 3, 16, 16])) for i in range(self.num_patches)]
        self.args = args


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for prompter in self.prompters:
            prompter.requires_grad = True

        for name, m in self.named_parameters():
            if self.args.model == 'dino':
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= self.args.grad_from_block:
                        m.requires_grad = True
                        
            elif self.args.model == 'clip':
                if 'transformer.resblocks' in name:
                    block_num = int(name.split('.')[2])
                    if block_num >= self.args.grad_from_block:
                        m.requires_grad = True


    def get_model_params(self, model):

        for prompter in self.prompters:
            prompter.requires_grad = False

        regularized = []
        not_regularized = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)

        param_model = [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
        
        for prompter in self.prompters:
            prompter.requires_grad = True

        regularized = []
        not_regularized = []

        for prompter in self.prompters:
            for name, param in prompter.named_parameters():
                if not param.requires_grad:
                    continue
                # we do not regularize biases nor Norm parameters
                if name.endswith(".bias") or len(param.shape) == 1:
                    not_regularized.append(param)
                else:
                    regularized.append(param)
        param_prompt = [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

        return param_model, param_prompt


    def attach(self, patch, propmter):
        self.fg_size = 16 - self.args.prompt_size * 2

        fg_in_patch = torch.zeros([1, 1, 3, self.fg_size, self.fg_size]).cuda()
        mask = F.pad(fg_in_patch, (self.args.prompt_size, self.args.prompt_size, self.args.prompt_size, self.args.prompt_size), "constant", 1)

        return patch + propmter.cuda() * mask


    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        # after CLS token, all before image patches
        x = self.prepare_tokens(x)  # (batch_size, 1 + n_patches, hidden_dim)
        B, n, c = x.shape

        cls_token = x[:, :1, :]
        embedded_patches = x[:, 1:, :].reshape(B, n-1, 3, 16, 16)
        new_patches = [self.attach(embedded_patches[:, i, :, :, :], self.prompters[i]) for i in range(n-1)]

        embedded_patches = torch.cat(new_patches, 1).reshape(B, n-1, c)

        x = torch.cat((
                cls_token,
                embedded_patches
            ), dim=1)
        # (batch_size, cls_token + n_patches, hidden_dim)

        return x
    
    def forward_deep_prompt(self, embedding_output):
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]

        for i in range(self.depth):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)

                hidden_states, weights = self.encoder.layer[i](hidden_states)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded

    def forward(self, x, return_all_patches=False):
        # this is the default version:
        embedding_output = self.incorporate_prompt(x)

        if self.args.use_deep:
            x = self.forward_deep_prompt(embedding_output)
        else:
            for blk in self.blocks:
                embedding_output = blk(embedding_output)
            x = self.norm(embedding_output)

        if return_all_patches:
            return x
        else:
            return x[:, 0]
