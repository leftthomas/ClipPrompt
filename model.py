import timm
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode='reflect'),
                                  nn.InstanceNorm2d(in_channels), nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode='reflect'),
                                  nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.conv(x)


class Generator(nn.Module):
    def __init__(self, in_channels=64, num_block=9):
        super(Generator, self).__init__()

        # in conv
        self.in_conv = nn.Sequential(nn.Conv2d(3, in_channels, 7, padding=3, padding_mode='reflect'),
                                     nn.InstanceNorm2d(in_channels), nn.ReLU(inplace=True))

        # down sample
        down_sample = []
        for _ in range(2):
            out_channels = in_channels * 2
            down_sample += [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                            nn.InstanceNorm2d(out_channels), nn.ReLU(inplace=True)]
            in_channels = out_channels
        self.down_sample = nn.Sequential(*down_sample)

        # conv blocks
        self.convs = nn.Sequential(*[ResidualBlock(in_channels) for _ in range(num_block)])

        # up sample
        up_sample = []
        for _ in range(2):
            out_channels = in_channels // 2
            up_sample += [nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(out_channels), nn.ReLU(inplace=True)]
            in_channels = out_channels
        self.up_sample = nn.Sequential(*up_sample)

        # out conv
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels, 3, 7, padding=3, padding_mode='reflect'), nn.Tanh())

    def forward(self, x):
        x = self.in_conv(x)
        x = self.down_sample(x)
        x = self.convs(x)
        x = self.up_sample(x)
        out = self.out_conv(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=64):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, in_channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels * 2, 4, stride=2, padding=1),
                                   nn.InstanceNorm2d(in_channels * 2), nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels * 4, 4, stride=2, padding=1),
                                   nn.InstanceNorm2d(in_channels * 4), nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, in_channels * 8, 4, padding=1),
                                   nn.InstanceNorm2d(in_channels * 8), nn.LeakyReLU(0.2, inplace=True))

        self.conv5 = nn.Conv2d(in_channels * 8, 1, 4, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.conv5(x)
        return out


class Extractor(nn.Module):
    def __init__(self, backbone_type, emb_dim, proxies):
        super(Extractor, self).__init__()

        # backbone
        model_name = 'resnet50' if backbone_type == 'resnet50' else 'vgg16'
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=emb_dim, global_pool='max')
        # self.proxies = nn.Parameter(torch.Tensor(len(proxies), emb_dim))
        # nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))
        # self.register_buffer('proxies', proxies)
        self.proxies = nn.Parameter(proxies)

    def forward(self, x):
        x = self.backbone(x)
        feature = F.normalize(x, dim=-1)
        classes = feature.matmul(F.normalize(self.proxies, dim=-1).t())
        return feature, classes


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()