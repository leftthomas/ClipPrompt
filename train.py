import clip
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import TripletMarginWithDistanceLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Model
from utils import DomainDataset, compute_metric, parse_args


# train for one epoch
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for img, pos, neg, label in train_bar:
        sketch_emb = net(img.cuda(), img_type='sketch')
        pos_emb = net(pos.cuda(), img_type='photo')
        neg_emb = net(neg.cuda(), img_type='photo')
        triplet_loss = triplet_criterion(sketch_emb, pos_emb, neg_emb)

        # normalized embeddings
        sketch_emb = F.normalize(sketch_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        # cosine similarity as logits
        logit_scale = 1.0 / net.clip_model.logit_scale
        logits_sketch = logit_scale * sketch_emb @ text_emb.t()
        logits_pos = logit_scale * pos_emb @ text_emb.t()

        cls_sketch_loss = cls_criterion(logits_sketch, label.cuda())
        cls_photo_loss = cls_criterion(logits_pos, label.cuda())
        loss = triplet_loss + (cls_sketch_loss + cls_photo_loss) * args.cls_weight
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        total_num += img.size(0)
        total_loss += loss.item() * img.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'
                                  .format(epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num


# val for one epoch
def val(net, data_loader):
    net.eval()
    vectors, domains, labels = [], [], []
    with torch.no_grad():
        for img, domain, label in tqdm(data_loader, desc='Feature extracting', dynamic_ncols=True):
            _, _, _, proj, _ = net(img.cuda())
            vectors.append(proj.cpu())
            domains.append(domain)
            labels.append(label)
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
    # args parse
    args = parse_args(mode='train')
    # data prepare
    train_data = DomainDataset(args.data_root, args.data_name, split='train')
    val_data = DomainDataset(args.data_root, args.data_name, split='val')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    # model and loss setup
    model = Model(args.prompt_num).cuda()
    triplet_criterion = TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
                                                      margin=args.triplet_margin)
    text = torch.cat([clip.tokenize('a photo of a {}'.format(train_data.names[c].replace('_', ' ')))
                      for c in sorted(train_data.names.keys())])
    with torch.no_grad():
        text_emb = F.normalize(model.clip_model.encode_text(text.cuda()), dim=-1)
    cls_criterion = CrossEntropyLoss()
    # optimizer config
    optimizer = Adam([{'params': model.sketch_encoder.parameters(), 'lr': args.encoder_lr},
                      {'params': model.photo_encoder.parameters(), 'lr': args.encoder_lr},
                      {'params': [model.sketch_prompt, model.photo_prompt], 'lr': args.prompt_lr}])
    # training loop
    results = {'train_loss': [], 'val_precise': [], 'P@100': [], 'P@200': [], 'mAP@200': [], 'mAP@all': []}
    save_name_pre = '{}_{}'.format(args.data_name, args.prompt_num)
    best_precise = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        val_precise, features = val(model, val_loader)
        results['val_precise'].append(val_precise * 100)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_results.csv'.format(args.save_root, save_name_pre), index_label='epoch')

        if val_precise > best_precise:
            best_precise = val_precise
            torch.save(model.state_dict(), '{}/{}_model.pth'.format(args.save_root, save_name_pre))