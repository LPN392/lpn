import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from net import IMAGENET_MEAN, IMAGENET_STD, build_model, save_arch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DEFAULT_DATA_ROOT = os.path.join(BASE_DIR, 'dataset', 'flowers')


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FileListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), target


def collect_samples(data_root: str):
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    class_to_idx = {name: i for i, name in enumerate(classes)}
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    by_class = defaultdict(list)
    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        for fn in os.listdir(cls_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                by_class[cls].append(os.path.join(cls_dir, fn))
    return classes, class_to_idx, by_class


def stratified_split(by_class, class_to_idx, val_ratio: float, seed: int):
    rng = random.Random(seed)
    train_samples, val_samples = [], []
    for cls, files in by_class.items():
        files = list(files)
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_ratio))
        val_files = files[:n_val]
        train_files = files[n_val:]
        idx = class_to_idx[cls]
        train_samples.extend((p, idx) for p in train_files)
        val_samples.extend((p, idx) for p in val_files)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def build_hard_class_multiplier(hard_report_path: str, recall_threshold: float, boost: float):
    if not hard_report_path or not os.path.isfile(hard_report_path):
        return {}
    with open(hard_report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    mul = {}
    for row in report.get('per_class', []):
        cls = row.get('class')
        recall = float(row.get('recall', 1.0))
        if cls and recall < recall_threshold:
            # 召回越低，放大系数越高，上限避免过拟合。
            factor = min(3.0, max(1.0, boost + (recall_threshold - recall) * 3.0))
            mul[cls] = factor

    # 将高频混淆对的两侧都纳入强化，帮助模型学会区分相近类别。
    for row in report.get('top_confusions', []):
        count = int(row.get('count', 0))
        true_cls = row.get('true')
        pred_cls = row.get('pred')
        if count < 2 or not true_cls or not pred_cls:
            continue

        pair_boost = min(3.0, 1.0 + (count * 0.25))
        mul[true_cls] = max(mul.get(true_cls, 1.0), pair_boost)
        mul[pred_cls] = max(mul.get(pred_cls, 1.0), min(2.5, 1.0 + (count * 0.15)))
    return mul


def apply_hard_class_oversampling(train_samples, classes, multiplier_map, seed: int):
    if not multiplier_map:
        return train_samples

    idx_to_class = {i: c for i, c in enumerate(classes)}
    by_idx = defaultdict(list)
    for path, idx in train_samples:
        by_idx[idx].append((path, idx))

    rng = random.Random(seed + 2026)
    boosted = list(train_samples)
    for idx, samples in by_idx.items():
        cls = idx_to_class[idx]
        factor = multiplier_map.get(cls, 1.0)
        if factor <= 1.0 or not samples:
            continue
        extra_n = int(round((factor - 1.0) * len(samples)))
        for _ in range(extra_n):
            boosted.append(rng.choice(samples))

    rng.shuffle(boosted)
    return boosted


def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss_sum += loss.item() * targets.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


def mixup_batch(images, targets, alpha: float):
    if alpha <= 0:
        return images, targets, targets, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    y_a, y_b = targets, targets[index]
    return mixed, y_a, y_b, lam


def set_backbone_trainable(model, trainable: bool):
    for name, p in model.named_parameters():
        if not name.startswith('fc.'):
            p.requires_grad = trainable


def main():
    parser = argparse.ArgumentParser(description='Flower classifier training')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT, help='Path to class folders')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--arch', default='resnet34', choices=['resnet18', 'resnet34'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--hard-report-path', default=os.path.join(MODELS_DIR, 'eval_report.json'))
    parser.add_argument('--hard-recall-threshold', type=float, default=0.5)
    parser.add_argument('--hard-boost', type=float, default=1.6)
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f'Dataset not found: {args.data_root}')

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    classes, class_to_idx, by_class = collect_samples(args.data_root)
    train_samples, val_samples = stratified_split(by_class, class_to_idx, args.val_ratio, args.seed)

    hard_multiplier = build_hard_class_multiplier(
        args.hard_report_path,
        recall_threshold=args.hard_recall_threshold,
        boost=args.hard_boost,
    )
    train_samples = apply_hard_class_oversampling(train_samples, classes, hard_multiplier, args.seed)

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError('Train/val split failed. Please check dataset files.')

    train_tf, val_tf = build_transforms(args.image_size)
    train_ds = FileListDataset(train_samples, train_tf)
    val_ds = FileListDataset(val_samples, val_tf)

    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(num_classes=len(classes), pretrained=True, arch=args.arch).to(device)
    set_backbone_trainable(model, trainable=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False)
    save_arch(args.arch)

    best_acc = -1.0
    best_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    history = []

    print(f'Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {len(classes)}')
    print(f'Device: {device}, AMP: {use_amp}, Arch: {args.arch}')
    if hard_multiplier:
        hard_txt = ', '.join([f'{k}x{v:.2f}' for k, v in sorted(hard_multiplier.items())])
        print(f'Hard-class oversampling enabled: {hard_txt}')

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - args.warmup_epochs),
                eta_min=args.min_lr,
            )
            print('Unfroze backbone for full fine-tuning.')

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                mixed_images, targets_a, targets_b, lam = mixup_batch(images, targets, args.mixup_alpha)
                logits = model(mixed_images)
                if args.mixup_alpha > 0:
                    loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
                else:
                    loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * targets.size(0)
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += targets.size(0)

            if args.log_interval > 0 and (step % args.log_interval == 0 or step == len(train_loader)):
                avg_loss = train_loss_sum / max(1, train_total)
                avg_acc = train_correct / max(1, train_total)
                print(
                    f'  Epoch {epoch:03d} Step {step:03d}/{len(train_loader)} '
                    f'| loss={avg_loss:.4f} acc={avg_acc * 100:.2f}%'
                )

        scheduler.step()

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
        })

        print(
            f'Epoch {epoch:03d}/{args.epochs} '
            f'| train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% '
            f'| val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}%'
        )

        improved = (val_acc > best_acc) or (abs(val_acc - best_acc) < 1e-6 and val_loss < best_loss)
        if improved:
            best_acc = val_acc
            best_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'model.pth'))
            print(f'  Saved new best model at epoch {epoch}.')
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f'Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).')
            break

    report = {
        'best_epoch': best_epoch,
        'best_val_acc': best_acc,
        'best_val_loss': best_loss,
        'epochs_ran': len(history),
        'settings': vars(args),
        'history': history,
    }
    with open(os.path.join(MODELS_DIR, 'train_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'Finished. Best epoch: {best_epoch}, best val acc: {best_acc * 100:.2f}%')


if __name__ == '__main__':
    main()
