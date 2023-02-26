import glob
import os
import random

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from metric import sake_metric


def get_transform(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class DomainDataset(Dataset):
    def __init__(self, data_root, data_name, split='train'):
        super(DomainDataset, self).__init__()

        images = []
        for classes in os.listdir(os.path.join(data_root, data_name, split, 'sketch')):
            sketches = glob.glob(os.path.join(data_root, data_name, split, 'sketch', str(classes), '*.jpg'))
            photos = glob.glob(os.path.join(data_root, data_name, split, 'photo', str(classes), '*.jpg'))
            # only consider the classes which photo images >= 400 for tuberlin dataset
            if len(photos) < 400 and data_name == 'tuberlin' and split == 'val':
                pass
            else:
                images += sketches
                # only append sketches for train
                if split == 'val':
                    images += photos
        self.images = sorted(images)
        self.transform = get_transform(split)

        self.domains, self.labels, self.classes = [], [], {}
        i = 0
        for img in self.images:
            domain, label = os.path.dirname(img).split('/')[-2:]
            self.domains.append(0 if domain == 'photo' else 1)
            if label not in self.classes:
                self.classes[label] = i
                i += 1
            self.labels.append(self.classes[label])
        # store photos for each class to easy sample for sketch in training period
        if split == 'train':
            self.refs = {}
            for key, value in self.classes.items():
                self.refs[value] = sorted(glob.glob(os.path.join(data_root, data_name, split, 'photo', key, '*.jpg')))

        self.split = split

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = self.transform(img)
        label = self.labels[index]
        if self.split == 'val':
            domain = self.domains[index]
            return img, domain, label
        else:
            ref = Image.open(random.choice(self.refs[label]))
            ref = self.transform(ref)
            return img, ref, label

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

