import copy

import torch
import torch.nn as nn

from prompt import load_clip


def unfreeze_ln(m):
    if isinstance(m, nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(True)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(True)


class Model(nn.Module):
    def __init__(self, prompt_num):
        super(Model, self).__init__()
        # backbone
        clip_model = load_clip('ViT-B/32')
        for param in clip_model.parameters():
            param.requires_grad_(False)
        visual = clip_model.visual
        visual.apply(unfreeze_ln)
        visual.proj.requires_grad_(True)

        self.sketch_encoder = visual
        self.photo_encoder = copy.deepcopy(visual)
        self.clip_model = clip_model

        # prompts
        self.sketch_prompt = nn.Parameter(torch.randn(prompt_num, self.sketch_encoder.class_embedding.shape[0]))
        self.photo_prompt = nn.Parameter(torch.randn(prompt_num, self.photo_encoder.class_embedding.shape[0]))

    def forward(self, img, img_type):
        if img_type == 'sketch':
            proj = self.sketch_encoder(img, self.sketch_prompt.expand(img.shape[0], -1, -1))
        else:
            proj = self.photo_encoder(img, self.photo_prompt.expand(img.shape[0], -1, -1))
        return proj