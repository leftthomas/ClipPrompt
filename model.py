import torch
import torch.nn as nn

from prompt import load_clip


def freeze_all_but_ln(m):
    if not isinstance(m, nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


class Model(nn.Module):
    def __init__(self, prompt_num, prompt_dim):
        super(Model, self).__init__()
        # backbone
        self.clip_model = load_clip('ViT-B/32')
        self.clip_model.apply(freeze_all_but_ln)

        # prompts
        self.sketch_prompt = nn.Parameter(torch.randn(prompt_num, prompt_dim))
        self.photo_prompt = nn.Parameter(torch.randn(prompt_num, prompt_dim))

    def forward(self, img, img_type='photo'):
        # text = torch.cat([clip.tokenize('a photo of a {}'.format(train_data.names[c].replace('_', ' ')))
        #                   for c in sorted(train_data.names.keys())])
        # with torch.no_grad():
        #     text_features = clip_model.encode_text(text.cuda())
        if img_type == 'sketch':
            proj = self.clip_model.encode_image(img, self.sketch_prompt.expand(img.shape[0], -1, -1))
        else:
            proj = self.clip_model.encode_image(img, self.photo_prompt.expand(img.shape[0], -1, -1))
        return proj