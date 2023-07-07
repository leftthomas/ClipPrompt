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
        self.sketch_encoder = load_clip('ViT-B/32').visual
        self.photo_encoder = load_clip('ViT-B/32').visual
        for param in self.sketch_encoder.parameters():
            param.requires_grad_(False)
        for param in self.photo_encoder.parameters():
            param.requires_grad_(False)
        self.sketch_encoder.apply(unfreeze_ln)
        self.photo_encoder.apply(unfreeze_ln)

        # prompts
        self.sketch_prompt = nn.Parameter(torch.randn(prompt_num, self.sketch_encoder.class_embedding.shape[0]))
        self.photo_prompt = nn.Parameter(torch.randn(prompt_num, self.photo_encoder.class_embedding.shape[0]))

    def forward(self, img, img_type='photo'):
        # text = torch.cat([clip.tokenize('a photo of a {}'.format(train_data.names[c].replace('_', ' ')))
        #                   for c in sorted(train_data.names.keys())])
        # with torch.no_grad():
        #     text_features = clip_model.encode_text(text.cuda())
        if img_type == 'sketch':
            proj = self.sketch_encoder.encode_image(img, self.sketch_prompt.expand(img.shape[0], -1, -1))
        else:
            proj = self.photo_encoder.encode_image(img, self.photo_prompt.expand(img.shape[0], -1, -1))
        return proj