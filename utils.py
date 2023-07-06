import argparse
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from metric import sake_metric


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
            neg_name = np.random.choice(self.refs[np.random.choice(list(self.classes.keys() - [self.names[label]]))])
            pos = self.transform(Image.open(pos_name))
            neg = self.transform(Image.open(neg_name))
            return img, pos, neg, label
        else:
            return img, domain, label

    def __len__(self):
        return len(self.images)


def compute_metric(vectors, domains, labels):
    acc = {}

    photo_vectors = vectors[domains == 0].numpy()
    sketch_vectors = vectors[domains == 1].numpy()
    photo_labels = labels[domains == 0].numpy()
    sketch_labels = labels[domains == 1].numpy()
    map_all, p_100 = sake_metric(photo_vectors, photo_labels, sketch_vectors, sketch_labels)
    map_200, p_200 = sake_metric(photo_vectors, photo_labels, sketch_vectors, sketch_labels,
                                 {'precision': 200, 'map': 200})

    acc['P@100'], acc['P@200'], acc['mAP@200'], acc['mAP@all'] = p_100, p_200, map_200, map_all
    # the mean value is chosen as the representative of precise
    acc['precise'] = (acc['P@100'] + acc['P@200'] + acc['mAP@200'] + acc['mAP@all']) / 4
    return acc


def parse_args(mode='train'):
    parser = argparse.ArgumentParser(description='Train/Test Model')
    # common args
    parser.add_argument('--data_root', default='/home/data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin'],
                        help='Dataset name')
    parser.add_argument('--prompt_dim', default=512, type=int, help='Prompt embedding dim')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    if mode == 'train':
        parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
        parser.add_argument('--epochs', default=10, type=int, help='Number of epochs over the model to train')
    else:
        parser.add_argument('--query_name', default='/home/data/sketchy/val/sketch/cow/n01887787_591-14.jpg', type=str,
                            help='Query image name')
        parser.add_argument('--num', default=8, type=int, help='Retrieval number')

    args = parser.parse_args()
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    return args