import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.backends import cudnn
from torch.utils.data.dataset import Dataset
from torchmetrics.functional.retrieval import retrieval_precision, retrieval_average_precision
from torchvision import transforms


def get_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, split='train'):
        super(DomainDataset, self).__init__()

        self.split = split

        images, self.refs = [], {}
        for classes in os.listdir(os.path.join(data_root, data_name, split, 'sketch')):
            sketches = glob.glob(os.path.join(data_root, data_name, split, 'sketch', str(classes), '*.jpg'))
            photos = glob.glob(os.path.join(data_root, data_name, split, 'photo', str(classes), '*.jpg'))
            # only consider the classes which photo images >= 400 for tuberlin dataset
            if len(photos) < 400 and data_name == 'tuberlin' and split == 'val':
                pass
            else:
                images += sketches
                if split == 'val':
                    images += photos
                else:
                    self.refs[str(classes)] = photos
        self.images = sorted(images)
        self.transform = get_transform()

        self.domains, self.labels, self.classes = [], [], {}
        i = 0
        for img in self.images:
            domain, label = os.path.dirname(img).split('/')[-2:]
            self.domains.append(0 if domain == 'photo' else 1)
            if label not in self.classes:
                self.classes[label] = i
                i += 1
            self.labels.append(self.classes[label])

        self.names = {}
        for key, value in self.classes.items():
            self.names[value] = key

    def __getitem__(self, index):
        img_name = self.images[index]
        domain = self.domains[index]
        label = self.labels[index]
        img = self.transform(Image.open(img_name))
        if self.split == 'train':
            pos_name = np.random.choice(self.refs[self.names[label]])
            remain_classes = sorted(set(self.classes.keys()) - {self.names[label]})
            neg_name = np.random.choice(self.refs[np.random.choice(remain_classes)])
            pos = self.transform(Image.open(pos_name))
            neg = self.transform(Image.open(neg_name))
            return img, pos, neg, label
        else:
            return img, domain, label

    def __len__(self):
        return len(self.images)


def compute_metric(vectors, domains, labels):
    acc = {}
    sketch_vectors, photo_vectors = vectors[domains == 1], vectors[domains == 0]
    sketch_labels, photo_labels = labels[domains == 1], labels[domains == 0]
    precs_100, precs_200, maps_200, maps_all = 0, 0, 0, 0
    for sketch_vector, sketch_label in zip(sketch_vectors, sketch_labels):
        sim = F.cosine_similarity(sketch_vector.unsqueeze(dim=0), photo_vectors).squeeze(dim=0)
        target = torch.zeros_like(sim, dtype=torch.bool)
        target[sketch_label == photo_labels] = True
        precs_100 += retrieval_precision(sim, target, top_k=100).item()
        precs_200 += retrieval_precision(sim, target, top_k=200).item()
        maps_200 += retrieval_average_precision(sim, target, top_k=200).item()
        maps_all += retrieval_average_precision(sim, target).item()

    prec_100 = precs_100 / sketch_vectors.shape[0]
    prec_200 = precs_200 / sketch_vectors.shape[0]
    map_200 = maps_200 / sketch_vectors.shape[0]
    map_all = maps_all / sketch_vectors.shape[0]

    acc['P@100'], acc['P@200'], acc['mAP@200'], acc['mAP@all'] = prec_100, prec_200, map_200, map_all
    # the mean value is chosen as the representative of precise
    acc['precise'] = (acc['P@100'] + acc['P@200'] + acc['mAP@200'] + acc['mAP@all']) / 4
    return acc


def parse_args():
    parser = argparse.ArgumentParser(description='Train/Test Model')
    # common args
    parser.add_argument('--data_root', default='/home/data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin'],
                        help='Dataset name')
    parser.add_argument('--prompt_num', default=3, type=int, help='Number of prompt embedding')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str, help='Mode of the script')

    # train args
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=60, type=int, help='Number of epochs over the model to train')
    parser.add_argument('--triplet_margin', default=0.3, type=float, help='Margin of triplet loss')
    parser.add_argument('--encoder_lr', default=1e-4, type=float, help='Learning rate of encoder')
    parser.add_argument('--prompt_lr', default=1e-3, type=float, help='Learning rate of prompt embedding')
    parser.add_argument('--cls_weight', default=0.5, type=float, help='Weight of classification loss')
    parser.add_argument('--seed', default=-1, type=int, help='random seed (-1 for no manual seed)')

    # test args
    parser.add_argument('--query_name', default='/home/data/sketchy/val/sketch/cow/n01887787_591-14.jpg', type=str,
                        help='Query image path')
    parser.add_argument('--retrieval_num', default=8, type=int, help='Number of retrieved images')

    args = parser.parse_args()
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    return args