import argparse
import json
import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from net import IMAGENET_MEAN, IMAGENET_STD, build_model, predict_probs


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


def make_val_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def top_confusions(conf_mat: torch.Tensor, classes, topn=12):
    rows = []
    n = conf_mat.size(0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = int(conf_mat[i, j].item())
            if v > 0:
                rows.append((v, classes[i], classes[j]))
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[:topn]


def main():
    parser = argparse.ArgumentParser(description='Evaluate model and export confusion analysis')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--model-path', default=os.path.join(MODELS_DIR, 'model.pth'))
    parser.add_argument('--labels-path', default=os.path.join(MODELS_DIR, 'labels.json'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--tta', action='store_true', help='Use TTA at evaluation')
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f'Model not found: {args.model_path}')
    if not os.path.isfile(args.labels_path):
        raise FileNotFoundError(f'Labels not found: {args.labels_path}')

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes_data, class_to_idx, by_class = collect_samples(args.data_root)
    with open(args.labels_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)

    if sorted(classes) != sorted(classes_data):
        raise RuntimeError('labels.json and dataset class folders mismatch.')

    _, val_samples = stratified_split(by_class, class_to_idx, args.val_ratio, args.seed)
    val_tf = make_val_transform(args.image_size)
    val_ds = FileListDataset(val_samples, val_tf)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    model = build_model(len(classes), pretrained=False).to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    conf_mat = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if args.tta:
                preds = []
                for i in range(images.size(0)):
                    # Rebuild PIL for reusing same TTA pipeline as online inference.
                    img = transforms.ToPILImage()(images[i].cpu() * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1))
                    probs = predict_probs(model, img, device, tta=True)
                    preds.append(int(torch.argmax(probs).item()))
                pred_idx = torch.tensor(preds, device=device)
            else:
                logits = model(images)
                pred_idx = logits.argmax(dim=1)

            for t, p in zip(targets, pred_idx):
                conf_mat[int(t.item()), int(p.item())] += 1

            correct += (pred_idx == targets).sum().item()
            total += targets.size(0)

    overall_acc = correct / max(1, total)

    per_class = []
    for i, name in enumerate(classes):
        tp = int(conf_mat[i, i].item())
        row_sum = int(conf_mat[i, :].sum().item())
        col_sum = int(conf_mat[:, i].sum().item())
        recall = tp / row_sum if row_sum > 0 else 0.0
        precision = tp / col_sum if col_sum > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append({
            'class': name,
            'support': row_sum,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

    per_class_sorted = sorted(per_class, key=lambda x: x['recall'])
    hard_pairs = top_confusions(conf_mat, classes, topn=15)

    report = {
        'overall_acc': overall_acc,
        'total': total,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'tta': bool(args.tta),
        'per_class': per_class,
        'worst_recall_classes': per_class_sorted[:10],
        'top_confusions': [
            {'count': c, 'true': t, 'pred': p}
            for c, t, p in hard_pairs
        ],
        'confusion_matrix': conf_mat.tolist(),
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'eval_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'Overall acc: {overall_acc * 100:.2f}% on {total} samples')
    print('Worst recall classes:')
    for row in per_class_sorted[:10]:
        print(f"  {row['class']}: recall={row['recall'] * 100:.2f}% support={row['support']}")
    print('Top confusions:')
    for c, t, p in hard_pairs[:10]:
        print(f'  {t} -> {p}: {c}')


if __name__ == '__main__':
    main()
