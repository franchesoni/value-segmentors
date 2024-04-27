import os
from math import ceil
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from torch import randint, float32, int64, Tensor
import torch.nn.functional as F


def resize(
    x: Tensor,
    size: any or None = None,
    scale_factor: list[float] or None = None,
    mode: str = "bilinear",
    align_corners: bool or None = False,
) -> Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


class ADE20KDataset(Dataset):
    def __init__(
        self,
        root="data/ADEChallengeData2016",
        split="training",
        transform=None,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.sample_paths = self.get_paths()

    def get_paths(self):
        sample_paths = []

        image_dir = os.path.join(self.root, "images", self.split)
        label_dir = os.path.join(self.root, "annotations", self.split)

        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, image_name.replace(".jpg", ".png"))
            sample_paths.append((image_path, label_path))

        return sample_paths

    def __getitem__(self, index):
        img_path, label_path = self.sample_paths[index]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.sample_paths)


img2tensor = Compose([ToImage(), ToDtype(float32, scale=True)])
label2tensor = Compose([ToImage(), ToDtype(int64)])


def resize_shortest_side(pilimage, pillabel, size=512):
    ratio = size / min(pilimage.size)
    new_size = (ceil(pilimage.size[0] * ratio), ceil(pilimage.size[1] * ratio))
    pilimage = pilimage.resize(new_size, resample=Image.Resampling.LANCZOS)
    pillabel = pillabel.resize(new_size, Image.NEAREST)
    return pilimage, pillabel, new_size


def val_transform(pilimage, pillabel):
    """resize shortest side to 512 and center crop a square of size 512x512"""
    pilimage, pillabel, new_size = resize_shortest_side(pilimage, pillabel)
    width_excess = new_size[0] - 512
    height_excess = new_size[1] - 512
    width_offset = width_excess // 2
    height_offset = height_excess // 2

    pilimage = pilimage.crop(
        (width_offset, height_offset, width_offset + 512, height_offset + 512)
    )
    pillabel = pillabel.crop(
        (width_offset, height_offset, width_offset + 512, height_offset + 512)
    )
    image = img2tensor(pilimage)
    label = label2tensor(pillabel) - 1
    return image, label


def train_transform(pilimage, pillabel):
    """resize shortest side to 512 and random crop a square of size 512x512"""
    pilimage, pillabel, new_size = resize_shortest_side(pilimage, pillabel)
    excess = max(new_size) - 512
    offset = int(randint(0, excess, (1,))) if excess else 0
    if new_size[0] > 512:
        width_offset = offset
        height_offset = 0
    else:
        width_offset = 0
        height_offset = offset

    pilimage = pilimage.crop(
        (width_offset, height_offset, width_offset + 512, height_offset + 512)
    )
    pillabel = pillabel.crop(
        (width_offset, height_offset, width_offset + 512, height_offset + 512)
    )
    image = img2tensor(pilimage)
    label = label2tensor(pillabel) - 1
    return image, label


