"""
To load the Imagenet Dataset (ILSVRC2012 version)
https://image-net.org/download.php
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


class ImagenetDataset(BaseDataset):
    def __init__(self, root: str | Path, split: str = "train", scale=1, min_size=64):
        super().__init__(root, split, scale)
        if split == "test":
            raise ValueError(f"Imagenet does not support test set")

        self.image_dir = self.root_dir / split

        # get paths to all image files needed
        self.image_files = list(self.image_dir.rglob("*.JPEG"))

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
        image_file_idx = index // 4
        image_file = self.image_files[image_file_idx]
        label = index % 4

        # read image
        image = Image.open(image_file).convert("RGB")
        image = self.transform(image)

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
    dotenv.load_dotenv()
    root = os.environ["IMAGENET_DATASET"]

    # try loading all the data
    start_timer = timeit.default_timer()
    dataset_train = ImagenetDataset(root, scale=0.4)
    ckpt = timeit.default_timer()
    print(f"Timefor loading training dataset: {ckpt-start_timer:.2f}s")

    dataset_val = ImagenetDataset(root, split="val")
    end_timer = timeit.default_timer()
    print(f"Timefor loading validation dataset: {end_timer-ckpt:.2f}s")

    print(f"{len(dataset_train)=}| {len(dataset_val)=}")

    # pry the data format
    image, label = dataset_train[0]
    print(image[0, :3, :3])
    print(image.shape, label)

    # try visualization
    ImagenetDataset.plot_image(image, 0, save_to="image0.png")
    image2, label2 = dataset_train[1]
    ImagenetDataset.plot_results([image, image2], [label, label2])


if __name__ == "__main__":
    _test()
