"""
To load the Cityscapes Dataset for panoptic segmentation
source: https://www.cityscapes-dataset.com/
github: https://github.com/mcordts/cityscapesScripts/tree/master
"""

from pathlib import Path
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image


class CityscapesDataset(data.Dataset):
    HEIGHT, WIDTH = 1024, 2048
    IMAGENET_MEAN = np.array((0.485, 0.456, 0.406))
    IMAGENET_STD = np.array((0.229, 0.224, 0.225))
    INPUT_CHANNELS = 3

    def __init__(self, root: str | Path, split: str = "train", scale=1):
        super().__init__()

        # process input
        if isinstance(root, str):
            self.root_dir = Path(root)
        else:
            self.root_dir = root
        assert self.root_dir.is_dir()
        assert scale > 0, "scale of images must be > 0"

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Expect split to be 'train', 'val' or 'test', got {split}"
            )
        self.image_dir = self.root_dir / "leftImg8bit" / split

        # get paths to all image files needed
        self.image_files = list(self.image_dir.rglob("*.png"))
        self.image_files.sort(key=lambda p: p.name)

        # generate (image_file, label) pairs
        self.samples = [(p, i) for p in self.image_files for i in range(4)]

        # transform for train images and labels/instance
        # the size is scaled accordingly
        self.scale = scale
        self.size = (int(self.HEIGHT * scale), int(self.WIDTH * scale))
        self.transform_image = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.size, antialias=None),
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD,
                    inplace=True,
                ),
            ]
        )

    def __getitem__(self, index):
        image_file, label = self.samples[index]

        # read image
        image = Image.open(image_file).convert("RGB")
        image = self.transform_image(image)

        # rotate image
        image = torch.rot90(image, k=label, dims=[-2, -1])

        return image, label

    def __len__(self):
        return len(self.samples)

    @classmethod
    def plot_image(cls, image: torch.Tensor, save_to="image.png"):
        # plot input training image

        # reverse the normalization used in training
        inverse_mean = -cls.IMAGENET_MEAN / cls.IMAGENET_STD
        inverse_std = 1 / cls.IMAGENET_STD
        transform = transforms.Compose(
            [
                transforms.Normalize(mean=inverse_mean, std=inverse_std),
                transforms.ToPILImage(),
            ]
        )
        image = transform(image)
        image.save(save_to)


def _test():
    import os
    from dotenv import load_dotenv

    # see where the dataset is
    load_dotenv()
    root = os.environ["CITYSCAPES_DATASET"]
    print(root)

    # try loading all the data
    dataset_train = CityscapesDataset(root, scale=0.4)
    dataset_val = CityscapesDataset(root, split="val")
    dataset_test = CityscapesDataset(root, split="test")
    print(f"{len(dataset_train)=}| {len(dataset_val)=}| {len(dataset_test)=}")

    # pry the data format
    image, label = dataset_train[0]
    print(image[0, :3, :3])
    print(image.shape, label)

    CityscapesDataset.plot_image(image, "image0.png")


if __name__ == "__main__":
    _test()
