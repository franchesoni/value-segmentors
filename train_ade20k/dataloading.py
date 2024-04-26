import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from torch import randint, float32, int64


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
        image = Image.open(img_path)
        label = Image.open(label_path)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.sample_paths)


img2tensor = Compose([ToImage(), ToDtype(float32, scale=True)])
label2tensor = Compose([ToImage(), ToDtype(int64)])


def train_transform(pilimage, pillabel):
    """resize shortest side to 512 and random crop a square of size 512x512"""
    ratio = 512 / min(pilimage.size)
    new_size = (int(pilimage.size[0] * ratio), int(pilimage.size[1] * ratio))
    pilimage = pilimage.resize(new_size, resample=Image.Resampling.LANCZOS)
    pillabel = pillabel.resize(new_size, Image.NEAREST)
    if new_size[0] > 512:
        width_excess = new_size[0] - 512
        width_offset = int(randint(width_excess, (1,)))
        height_offset = 0
    else:
        height_excess = new_size[1] - 512
        height_offset = int(randint(height_excess, (1,)))
        width_offset = 0

    pilimage = pilimage.crop(
        (width_offset, height_offset, width_offset + 512, height_offset + 512)
    )
    pillabel = pillabel.crop(
        (width_offset, height_offset, width_offset + 512, height_offset + 512)
    )
    image = img2tensor(pilimage)
    label = label2tensor(pillabel)
    return image, label
