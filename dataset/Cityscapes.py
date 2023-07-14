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
    CHANNELS, HEIGHT, WIDTH = 3, 1024, 2048
    IMAGENET_MEAN = np.array((0.485, 0.456, 0.406))
    IMAGENET_STD = np.array((0.229, 0.224, 0.225))

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
            raise ValueError(f"Expect split to be 'train', 'val' or 'test', got {split}")
        self.image_dir = self.root_dir / "leftImg8bit" / split

        # get paths to all image files needed
        self.image_files = list(self.image_dir.rglob("*.png"))
        self.image_files.sort(key=lambda p: p.name)

        # generate (image_file, label) pairs
        self.samples = [(p, i) for p in self.image_files for i in range(4)]

        # transform for train images and labels/instance
        # the size is scaled to match the longest side (width)
        self.scale = scale
        self.size = (int(self.WIDTH * scale), int(self.WIDTH * scale))
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.size, antialias=None),
            transforms.Normalize(
                mean=self.IMAGENET_MEAN,
                std=self.IMAGENET_STD,
                inplace=True,
            ),
        ])

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
    def plot_image(cls, image: torch.Tensor, num_rotation=0, save_to="image.png") -> Image.Image:
        # rotate image
        image = torch.rot90(image, k=num_rotation, dims=[-2, -1])

        # reverse the normalization used in training
        inverse_mean = -cls.IMAGENET_MEAN / cls.IMAGENET_STD
        inverse_std = 1 / cls.IMAGENET_STD
        transform = transforms.Compose([
            transforms.Normalize(mean=inverse_mean, std=inverse_std),
            transforms.ToPILImage(),
        ])
        image = transform(image)

        if save_to is not None:
            image.save(save_to)
        return image

    @classmethod
    def plot_results(cls,
                     images: torch.Tensor | list[torch.Tensor],
                     labels: list[int],
                     size=256,
                     save_to="samples.png"):
        """
        Visualize model results by giving input images and predictions of number of rotations
        """
        canvas = Image.new('RGB', (size * len(images), size * 2))

        for i, (image, label) in enumerate(zip(images, labels)):
            # paste input image in upper half
            x, y = size * i, 0
            input_image = cls.plot_image(image, 0, save_to=None)
            resized_image, (offset_x, offset_y) = cls._plot_image_in_square(input_image, size)

            paste_x = x + offset_x
            paste_y = y + offset_y
            canvas.paste(resized_image, (paste_x, paste_y))

            # paste predicted image in lower half
            y += size
            num_rotation = 4 - label
            rotated_image = cls.plot_image(image, num_rotation, save_to=None)
            resized_image, (offset_x, offset_y) = cls._plot_image_in_square(rotated_image, size)

            paste_x = x + offset_x
            paste_y = y + offset_y
            canvas.paste(resized_image, (paste_x, paste_y))

        if save_to is not None:
            canvas.save(save_to)
        return canvas

    @classmethod
    def _plot_image_in_square(cls, image: Image.Image, size: int):
        # scale input_image and calculate offset in the square
        width, height = image.size
        max_side = max(width, height)
        new_size = (int(width * size // max_side), int(height * size // max_side))
        resized_image = image.resize(new_size)

        offset_x = size // 2 - resized_image.width // 2
        offset_y = size // 2 - resized_image.height // 2
        return resized_image, (offset_x, offset_y)


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
