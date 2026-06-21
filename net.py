"""花卉分类网络：训练与推理共用。"""
import json
import os

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms
from typing import Optional
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAD_RGB = tuple(int(c * 255) for c in IMAGENET_MEAN)
DEFAULT_ARCH = 'resnet34'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'models', 'config.json')
TTA_SIZES = (224, 256, 288)

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _resize_shorter_side(img: Image.Image, shorter: int) -> Image.Image:
    w, h = img.size
    if min(w, h) == shorter:
        return img
    scale = shorter / min(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.BICUBIC)


def center_crop(img: Image.Image, size: int = 224) -> Image.Image:
    img = ImageOps.exif_transpose(img.convert('RGB'))
    img = _resize_shorter_side(img, 256)
    w, h = img.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    return img.crop((left, top, left + size, top + size))


def load_arch():
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, encoding='utf-8') as f:
            return json.load(f).get('arch', DEFAULT_ARCH)
    return 'resnet18'


def save_arch(arch: str):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({'arch': arch}, f)


def build_model(num_classes: int, pretrained: bool = False, arch: Optional[str] = None):
    arch = arch or load_arch()
    if arch == 'resnet34':
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
    else:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
    in_f = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_f, num_classes),
    )
    return backbone


def build_resnet18(num_classes: int, pretrained: bool = False):
    return build_model(num_classes, pretrained, 'resnet18')


def letterbox(img: Image.Image, size: int = 224, fill=PAD_RGB) -> Image.Image:
    img = ImageOps.exif_transpose(img.convert('RGB'))
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def predict_probs(model, image: Image.Image, device, tta: bool = True):
    """融合 letterbox 与 center-crop 的加权 TTA，减少单一预处理造成的偏差。"""
    model.eval()
    image = ImageOps.exif_transpose(image.convert('RGB'))
    logits_sum = None
    weight_sum = 0.0

    # 视角定义: (kind, size, flip, weight)
    if tta:
        views = [
            ('letterbox', 224, False, 1.0),
            ('letterbox', 224, True, 1.0),
            ('letterbox', 256, False, 0.9),
            ('letterbox', 256, True, 0.9),
            ('center_crop', 224, False, 1.1),
            ('center_crop', 224, True, 1.1),
        ]
    else:
        views = [('letterbox', 224, False, 1.0)]

    with torch.no_grad():
        for kind, size, flip, weight in views:
            view = image.transpose(Image.FLIP_LEFT_RIGHT) if flip else image
            if kind == 'center_crop':
                prepared = center_crop(view, size)
            else:
                prepared = letterbox(view, size)
            tensor = _to_tensor(prepared).unsqueeze(0).to(device)
            out = model(tensor)
            logits_sum = out * weight if logits_sum is None else logits_sum + out * weight
            weight_sum += weight

        return torch.softmax(logits_sum / max(weight_sum, 1e-6), dim=1)[0]


def topk_probs(probabilities: torch.Tensor, class_names: list[str], k: int = 3):
    k = max(1, min(k, probabilities.numel()))
    vals, idxs = torch.topk(probabilities, k)
    return [
        {
            'index': int(i.item()),
            'name': class_names[int(i.item())],
            'prob': float(v.item()),
        }
        for v, i in zip(vals, idxs)
    ]
