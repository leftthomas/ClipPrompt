import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from PIL import Image

from model import Model
from utils import get_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Model')
    parser.add_argument('--vis_name', default='/home/data/sketchy/val/photo/helicopter/ext_5.jpg', type=str,
                        help='Visual image name')
    parser.add_argument('--model_name', default='result/sketchy_resnet50_512_model.pth', type=str,
                        help='Model name')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    opt = parser.parse_args()

    vis_name, model_name, save_root = opt.vis_name, opt.model_name, opt.save_root
    backbone_type, proj_dim = model_name.split('/')[-1].split('_')[1:3]
    size = (224, 224)

    vis_image = Image.open(vis_name).resize(size, resample=Image.BICUBIC)
    model = Model(backbone_type, int(proj_dim))
    model.load_state_dict(torch.load(model_name, map_location='cpu'))
    model = model.cuda()
    model.eval()

    result_path = '{}/{}'.format(save_root, vis_name.split('/')[-1].split('.')[0])
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    vis_image.save('{}/vis.jpg'.format(result_path))

    vis_image = get_transform()(Image.open(vis_name)).unsqueeze(dim=0).cuda()
    with torch.no_grad():
        block_1_atte, block_2_atte, block_3_atte, _ = model(vis_image)
        block_1_atte = cv2.resize(block_1_atte.sum(dim=1).squeeze().cpu().numpy(), size, interpolation=cv2.INTER_CUBIC)
        block_2_atte = cv2.resize(block_2_atte.sum(dim=1).squeeze().cpu().numpy(), size, interpolation=cv2.INTER_CUBIC)
        block_3_atte = cv2.resize(block_3_atte.sum(dim=1).squeeze().cpu().numpy(), size, interpolation=cv2.INTER_CUBIC)
        fused_atte = block_1_atte + block_2_atte + block_3_atte

    img = cv2.resize(cv2.imread(vis_name), size, interpolation=cv2.INTER_CUBIC)
    for i, atte in enumerate([block_1_atte, block_2_atte, block_3_atte, fused_atte]):
        atte -= atte.min()
        if atte.max() != 0:
            atte /= atte.max()
        heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * atte), cv2.COLORMAP_JET))
        cam = heat_map + np.float32(img)
        cam -= cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        cam = np.uint8(255 * cam)
        cv2.imwrite('{}/{}_atte.png'.format(result_path, i + 1), cam)