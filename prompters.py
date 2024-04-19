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
