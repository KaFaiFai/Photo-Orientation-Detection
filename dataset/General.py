"""
A general dataset type to load image folders
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


class GeneralDataset(BaseDataset):
    IMAGE_EXTENSION = (".jpg", ".jpeg", ".png")
    min_size = 33

    def __init__(self, root: str | Path, split: str = "train", scale=1):
        """
        Expected folder structure:
        root
        |--     train
        |       |--     folder1
        |               |--     **
        |                       |-- image.jpg
        |--     val
        |       |--     folder1
        |               |--     **
        |                       |-- image.jpeg
        |--     test
                |--     folder1
                        |--     **
                                |-- image.png

        """
        super().__init__(root, split, scale)

        self.image_dir = self.root_dir / split

        # get paths to all image files needed
        self.image_files = []
        for file in self.image_dir.rglob("*.*"):
            if file.suffix.lower() in self.IMAGE_EXTENSION:
                self.image_files.append(file)
        self.image_files.sort(key=lambda p: p.name)

        # transform for train images and labels/instance
        self.transform_image = transforms.Compose(
            [
                self.rescale(self.scale, min_size=self.min_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD,
                    inplace=True,
                ),
            ]
        )

    def __getitem__(self, index):
        image_file_idx = index // 4
        image_file = self.image_files[image_file_idx]
        label = index % 4

        # read image
        image = Image.open(image_file).convert("RGB")
        image = self.transform_image(image)

        # rotate image
        image = torch.rot90(image, k=label, dims=[-2, -1])

        return image, label

    def __len__(self):
        return len(self.image_files) * 4


def _test():
    import os
    import dotenv
    import timeit

    # see where the dataset is
    root = "training_data/variable_size"

    # try loading all the data
    dataset_train = GeneralDataset(root, scale=0.4)
    dataset_val = GeneralDataset(root, split="val")
    print(f"{len(dataset_train)=}| {len(dataset_val)=}")

    # pry the data format
    image, label = dataset_train[0]
    print(image[0, :3, :3])
    print(image.shape, label)

    # try visualization
    GeneralDataset.plot_image(image, 0, save_to="image0.png")
    image2, label2 = dataset_train[1]
    GeneralDataset.plot_results([image, image2], [label, label2])


if __name__ == "__main__":
    _test()
