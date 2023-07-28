"""
To load the Cityscapes Dataset
source: https://www.cityscapes-dataset.com/
github: https://github.com/mcordts/cityscapesScripts/tree/master
"""

from pathlib import Path
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image
import sys

sys.path.append(str((Path(__file__) / "..").resolve()))
from BaseDataset import BaseDataset


class CityscapesDataset(BaseDataset):
    def __init__(self, root: str | Path, split: str = "train", scale=1, min_size=64):
        super().__init__(root, split, scale)

        self.image_dir = self.root_dir / "leftImg8bit" / split

        # get paths to all image files needed
        self.image_files = list(self.image_dir.rglob("*.png"))
        self.image_files.sort(key=lambda p: p.name)

        # generate (image_file, label) pairs
        self.samples = [(p, i) for p in self.image_files for i in range(4)]

        # transform for train images and labels/instance
        self.transform = transforms.Compose(
            [
                self.rescale(scale, min_size=min_size),
                transforms.ToTensor(),
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
        image = self.transform(image)

        # rotate image
        image = torch.rot90(image, k=label, dims=[-2, -1])

        return image, label

    def __len__(self):
        return len(self.samples)


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

    # try visualization
    CityscapesDataset.plot_image(image, 0, save_to="image0.png")
    image2, label2 = dataset_train[1]
    CityscapesDataset.plot_results([image, image2], [label, label2])


if __name__ == "__main__":
    _test()
