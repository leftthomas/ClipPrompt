import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAM(nn.Module):
    def __init__(self, eps=1e-4):
        super(SimAM, self).__init__()
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = d / (4 * (d.sum(dim=[2, 3], keepdim=True) / n + self.eps)) + 0.5
        return torch.sigmoid(y)


class EnergyAttention(nn.Module):
    def __init__(self, low_dim, high_dim):
        super(EnergyAttention, self).__init__()
        self.conv = nn.Conv2d(high_dim, low_dim, kernel_size=3, padding=1)
        self.atte = SimAM()

    def forward(self, low_feat, high_feat):
        high_feat = F.interpolate(self.conv(high_feat), low_feat.size()[-2:], mode='bilinear', align_corners=False)
        atte = self.atte(torch.relu(low_feat + high_feat))
        low_feat = atte * low_feat
        return atte, low_feat


class Model(nn.Module):
    def __init__(self, backbone_type, proj_dim, proxies):
        super(Model, self).__init__()

        # backbone
        self.backbone = timm.create_model('seresnet50' if backbone_type == 'resnet50' else 'vgg16_bn',
                                          features_only=True, out_indices=(2, 3, 4), pretrained=True)
        dims = [512, 1024, 2048] if backbone_type == 'resnet50' else [256, 512, 512]

        # atte
        self.energy_1 = EnergyAttention(dims[0], dims[2])
        self.energy_2 = EnergyAttention(dims[1], dims[2])

        # proj
        self.proj = nn.Linear(sum(dims), proj_dim)

        # proxy
        # self.proxies = nn.Parameter(torch.Tensor(len(proxies), proj_dim))
        # nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))
        self.register_buffer('proxies', proxies)
        # self.proxies = nn.Parameter(proxies)

    def forward(self, img):
        block_1_feat, block_2_feat, block_3_feat = self.backbone(img)

        block_1_atte, block_1_feat = self.energy_1(block_1_feat, block_3_feat)
        block_2_atte, block_2_feat = self.energy_2(block_2_feat, block_3_feat)
        block_3_atte = torch.sigmoid(block_3_feat)

        block_1_feat = torch.flatten(F.adaptive_max_pool2d(block_1_feat, (1, 1)), start_dim=1)
        block_2_feat = torch.flatten(F.adaptive_max_pool2d(block_2_feat, (1, 1)), start_dim=1)
        block_3_feat = torch.flatten(F.adaptive_max_pool2d(block_3_feat, (1, 1)), start_dim=1)

        feat = torch.cat((block_1_feat, block_2_feat, block_3_feat), dim=-1)
        proj = F.normalize(self.proj(feat), dim=-1)
        classes = proj.matmul(F.normalize(self.proxies, dim=-1).t())
        return block_1_atte, block_2_atte, block_3_atte, proj, classes