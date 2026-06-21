"""Targeted fine-tuning: oversample/confine to specific classes while keeping others.

Usage examples:
  python train_targeted.py --focus rose,lotus --boost sunflower --epochs 8 --batch-size 16
  python train_targeted.py --focus-file focus.txt --factor 5 --resume models/model.pth
"""
import os
import json
import argparse
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets

from net import build_model


def parse_args():
    p = argparse.ArgumentParser(description='Targeted fine-tune for confusing classes')
    p.add_argument('--data-root', default=os.path.join(os.path.dirname(__file__), 'dataset', 'flowers'))
    p.add_argument('--models-dir', default=os.path.join(os.path.dirname(__file__), 'models'))
    p.add_argument('--focus', default='', help='comma separated class names (en) to oversample')
    p.add_argument('--focus-file', default='', help='file with one class per line')
    p.add_argument('--boost', default='', help='comma separated class names to gently boost')
    p.add_argument('--factor', type=float, default=3.0, help='oversample factor for focus classes')
    p.add_argument('--boost-factor', type=float, default=1.5, help='boost factor for boosted classes')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--resume', default='', help='checkpoint to load as initialization')
    p.add_argument('--arch', default='resnet34')
    return p.parse_args()


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(320),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return train_tf


def main():
    args = parse_args()

    focus = set()
    if args.focus:
        focus.update([s.strip() for s in args.focus.split(',') if s.strip()])
    if args.focus_file and os.path.exists(args.focus_file):
        with open(args.focus_file, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s: focus.add(s)

    boost = set([s.strip() for s in args.boost.split(',') if s.strip()])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    train_tf = build_transforms()
    train_dataset = datasets.ImageFolder(args.data_root, transform=train_tf)
    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    # compute per-sample weights
    counts = Counter([y for _, y in train_dataset.samples])
    weights = []
    for _, label in train_dataset.samples:
        cls_name = classes[label]
        w = 1.0
        if cls_name in focus:
            w *= args.factor
        if cls_name in boost:
            w *= args.boost_factor
        # inverse by count to favor rare classes a bit
        w = w / max(1.0, counts[label])
        weights.append(w)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers)

    # model
    with open(os.path.join(args.models_dir, 'labels.json'), 'r', encoding='utf-8') as f:
        labels = json.load(f)

    net = build_model(len(labels), arch=args.arch)
    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location='cpu')
        try:
            net.load_state_dict(state)
        except Exception:
            # try weights-only
            net.load_state_dict(state)
    net.to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = 1e9
    os.makedirs(args.models_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        net.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for imgs, labels_idx in loader:
            imgs = imgs.to(device)
            labels_idx = labels_idx.to(device)
            opt.zero_grad()
            out = net(imgs)
            loss = criterion(out, labels_idx)
            loss.backward()
            opt.step()

            total_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels_idx).sum().item()
            total += imgs.size(0)

        avg_loss = total_loss / total
        acc = correct / total * 100.0
        print(f'Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  acc={acc:.2f}%')

        checkpoint = os.path.join(args.models_dir, f'targeted_epoch{epoch}.pth')
        torch.save(net.state_dict(), checkpoint)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), os.path.join(args.models_dir, 'model_targeted_best.pth'))

    print('Fine-tune finished. Best loss:', best_loss)


if __name__ == '__main__':
    main()
