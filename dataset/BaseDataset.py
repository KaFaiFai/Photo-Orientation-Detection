"""
Base of all custom datasets in this repo
Contains useful image preprocessing function and tests
"""

from pathlib import Path
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image


class BaseDataset(data.Dataset):
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

        self.scale = scale
        assert scale > 0, "scale of images must be > 0"

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Expect split to be 'train', 'val' or 'test', got {split}")

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """
        Return an image of shape (channel, height, width) and a class index
        """
        raise NotImplementedError("Please implement this method")

    @classmethod
    def rescale(cls, scale=1):
        """
        return a function that rescale image tensor by scale
        """

        def _rescale(image: Image.Image):
            width, height = image.size
            new_size = (int(width * scale), int(height * scale))
            resized_image = image.resize(new_size)
            return resized_image

        return _rescale

    @classmethod
    def plot_image(cls, image: torch.Tensor, num_rotation=0, save_to="image.png") -> Image.Image:
        """
        Draw a rotated image and save it
        """
        # rotate image
        image = torch.rot90(image, k=num_rotation, dims=[-2, -1])

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

        if save_to is not None:
            image.save(save_to)
        return image

    @classmethod
    def plot_results(
        cls,
        images: torch.Tensor | list[torch.Tensor],
        labels: list[int],
        size=256,
        save_to="samples.png",
    ):
        """
        Visualize model results by giving input images and predictions of number of rotations
        """
        canvas = Image.new("RGB", (size * len(images), size * 2))

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
