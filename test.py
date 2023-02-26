import argparse
import os
import shutil

import torch
from PIL import Image, ImageDraw

from utils import DomainDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('--data_root', default='/home/data', type=str, help='Datasets root path')
    parser.add_argument('--query_name', default='/home/data/sketchy/val/sketch/cow/n01887787_591-14.jpg', type=str,
                        help='Query image name')
    parser.add_argument('--data_base', default='result/sketchy_resnet50_512_vectors.pth', type=str,
                        help='Queried database')
    parser.add_argument('--num', default=5, type=int, help='Retrieval number')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    opt = parser.parse_args()

    data_root, query_name, data_base, retrieval_num = opt.data_root, opt.query_name, opt.data_base, opt.num
    save_root, data_name = opt.save_root, data_base.split('/')[-1].split('_')[0]

    vectors = torch.load(data_base)
    val_data = DomainDataset(data_root, data_name, split='val')

    if query_name not in val_data.images:
        raise FileNotFoundError('{} not found'.format(query_name))
    query_index = val_data.images.index(query_name)
    query_image = Image.open(query_name).resize((224, 224), resample=Image.BILINEAR)
    query_label = val_data.labels[query_index]
    query_feature = vectors[query_index]

    gallery_images, gallery_labels = [], []
    for i, domain in enumerate(val_data.domains):
        if domain == 0:
            gallery_images.append(val_data.images[i])
            gallery_labels.append(val_data.labels[i])
    gallery_features = vectors[torch.tensor(val_data.domains) == 0]

    sim_matrix = query_feature.unsqueeze(0).mm(gallery_features.t()).squeeze()
    idx = sim_matrix.topk(k=retrieval_num, dim=-1)[1]

    result_path = '{}/{}'.format(save_root, query_name.split('/')[-1].split('.')[0])
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)
    query_image.save('{}/query.jpg'.format(result_path))
    for num, index in enumerate(idx):
        retrieval_image = Image.open(gallery_images[index.item()]).resize((224, 224), resample=Image.BILINEAR)
        draw = ImageDraw.Draw(retrieval_image)
        retrieval_label = gallery_labels[index.item()]
        retrieval_status = retrieval_label == query_label
        retrieval_sim = sim_matrix[index.item()].item()
        if retrieval_status:
            draw.rectangle((0, 0, 223, 223), outline='green', width=8)
        else:
            draw.rectangle((0, 0, 223, 223), outline='red', width=8)
        retrieval_image.save('{}/retrieval_{}_{}.jpg'.format(result_path, num + 1, '%.4f' % retrieval_sim))