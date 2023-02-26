import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import clip
from model import Extractor, Discriminator, Generator, set_bn_eval
from utils import DomainDataset, compute_metric

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(backbone, data_loader):
    backbone.train()
    # fix bn on backbone
    backbone.apply(set_bn_eval)
    generator.train()
    discriminator.train()
    total_extractor_loss, total_generator_loss, total_identity_loss, total_discriminator_loss = 0.0, 0.0, 0.0, 0.0
    total_num, train_bar = 0, tqdm(data_loader, dynamic_ncols=True)
    for sketch, photo, label in train_bar:
        sketch, photo, label = sketch.cuda(), photo.cuda(), label.cuda()

        optimizer_generator.zero_grad()
        optimizer_extractor.zero_grad()

        # generator #
        fake = generator(sketch)
        pred_fake = discriminator(fake)

        # generator loss
        target_fake = torch.ones(pred_fake.size(), device=pred_fake.device)
        gg_loss = adversarial_criterion(pred_fake, target_fake)
        total_generator_loss += gg_loss.item() * sketch.size(0)
        # identity loss
        ii_loss = identity_criterion(generator(photo), photo)
        total_identity_loss += ii_loss.item() * sketch.size(0)

        # extractor #
        sketch_proj, sketch_class = backbone(sketch)
        photo_proj, photo_class = backbone(photo)
        fake_proj, fake_class = backbone(fake)

        # extractor loss
        class_loss = (class_criterion(sketch_class/0.05, label) + class_criterion(photo_class/0.05, label) +
                      class_criterion(fake_class/0.05, label)) / 3
        total_extractor_loss += class_loss.item() * sketch.size(0)

        (gg_loss + 0.1 * ii_loss + 10 * class_loss).backward()

        optimizer_generator.step()
        optimizer_extractor.step()

        # discriminator loss #
        optimizer_discriminator.zero_grad()
        pred_photo = discriminator(photo)
        target_photo = torch.ones(pred_photo.size(), device=pred_photo.device)
        pred_fake = discriminator(fake.detach())
        target_fake = torch.zeros(pred_fake.size(), device=pred_fake.device)
        adversarial_loss = (adversarial_criterion(pred_photo, target_photo) +
                            adversarial_criterion(pred_fake, target_fake)) / 2
        total_discriminator_loss += adversarial_loss.item() * sketch.size(0)

        adversarial_loss.backward()
        optimizer_discriminator.step()

        total_num += sketch.size(0)

        e_loss = total_extractor_loss / total_num
        g_loss = total_generator_loss / total_num
        i_loss = total_identity_loss / total_num
        d_loss = total_discriminator_loss / total_num
        train_bar.set_description('Train Epoch: [{}/{}] E-Loss: {:.4f} G-Loss: {:.4f} I-Loss: {:.4f} D-Loss: {:.4f}'
                                  .format(epoch, epochs, e_loss, g_loss, i_loss, d_loss))

    return e_loss, g_loss, i_loss, d_loss


# val for one epoch
def val(backbone, encoder, data_loader):
    backbone.eval()
    encoder.eval()
    vectors, domains, labels = [], [], []
    with torch.no_grad():
        for img, domain, label in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            img = img.cuda()
            photo = img[domain == 0]
            sketch = img[domain == 1]
            photo_emb, _ = backbone(photo)
            sketch_emb, _ = backbone(encoder(sketch))
            emb = torch.cat((photo_emb, sketch_emb), dim=0)
            vectors.append(emb.cpu())
            label = torch.cat((label[domain == 0], label[domain == 1]), dim=0)
            labels.append(label)
            domain = torch.cat((domain[domain == 0], domain[domain == 1]), dim=0)
            domains.append(domain)
        vectors = torch.cat(vectors, dim=0)
        domains = torch.cat(domains, dim=0)
        labels = torch.cat(labels, dim=0)
        acc = compute_metric(vectors, domains, labels)
        results['P@100'].append(acc['P@100'] * 100)
        results['P@200'].append(acc['P@200'] * 100)
        results['mAP@200'].append(acc['mAP@200'] * 100)
        results['mAP@all'].append(acc['mAP@all'] * 100)
        print('Val Epoch: [{}/{}] | P@100:{:.1f}% | P@200:{:.1f}% | mAP@200:{:.1f}% | mAP@all:{:.1f}%'
              .format(epoch, epochs, acc['P@100'] * 100, acc['P@200'] * 100, acc['mAP@200'] * 100,
                      acc['mAP@all'] * 100))
    return acc['precise'], vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='/home/data', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='sketchy', type=str, choices=['sketchy', 'tuberlin'],
                        help='Dataset name')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'vgg16'],
                        help='Backbone type')
    parser.add_argument('--emb_dim', default=512, type=int, help='Embedding dim')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs over the model to train')
    parser.add_argument('--warmup', default=1, type=int, help='Number of warmups over the extractor to train')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    # args parse
    args = parser.parse_args()
    data_root, data_name, backbone_type, emb_dim = args.data_root, args.data_name, args.backbone_type, args.emb_dim
    batch_size, epochs, warmup, save_root = args.batch_size, args.epochs, args.warmup, args.save_root

    # data prepare
    train_data = DomainDataset(data_root, data_name, split='train')
    val_data = DomainDataset(data_root, data_name, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size // 2, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # model define
    extractor = Extractor(backbone_type, emb_dim, len(train_data.classes)).cuda()
    generator = Generator(in_channels=8, num_block=8).cuda()
    discriminator = Discriminator(in_channels=8).cuda()

    # loss setup
    class_criterion = nn.CrossEntropyLoss()
    adversarial_criterion = nn.MSELoss()
    identity_criterion = nn.L1Loss()
    # optimizer config
    optimizer_extractor = Adam([{'params': extractor.parameters()}], lr=1e-5)
    optimizer_generator = Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    optimizer_discriminator = Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # training loop
    results = {'extractor_loss': [], 'generator_loss': [], 'identity_loss': [], 'discriminator_loss': [],
               'precise': [], 'P@100': [], 'P@200': [], 'mAP@200': [], 'mAP@all': []}
    save_name_pre = '{}_{}_{}'.format(data_name, backbone_type, emb_dim)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    best_precise = 0.0
    for epoch in range(1, epochs + 1):

        # warmup, not update the parameters of extractor, except the final fc layer
        for param in list(extractor.backbone.parameters())[:-2]:
            param.requires_grad = False if epoch <= warmup else True

        extractor_loss, generator_loss, identity_loss, discriminator_loss = train(extractor, train_loader)
        results['extractor_loss'].append(extractor_loss)
        results['generator_loss'].append(generator_loss)
        results['identity_loss'].append(identity_loss)
        results['discriminator_loss'].append(discriminator_loss)
        precise, features = val(extractor, generator, val_loader)
        results['precise'].append(precise * 100)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(save_root, save_name_pre), index_label='epoch')

        if precise > best_precise:
            best_precise = precise
            torch.save(extractor.state_dict(), '{}/{}_extractor.pth'.format(save_root, save_name_pre))
            torch.save(generator.state_dict(), '{}/{}_generator.pth'.format(save_root, save_name_pre))
            torch.save(discriminator.state_dict(), '{}/{}_discriminator.pth'.format(save_root, save_name_pre))
            torch.save(class_criterion.state_dict(), '{}/{}_proxies.pth'.format(save_root, save_name_pre))
            torch.save(features, '{}/{}_vectors.pth'.format(save_root, save_name_pre))