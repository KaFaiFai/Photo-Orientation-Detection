import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def DEPvariable_size_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # target = torch.LongTensor(target)
    # return [data, target]
    return batch


def DEPidentity_collate(batch):
    return batch


def pad_collate(batch):
    images: list[Tensor] = [item[0] for item in batch]
    max_height = max([image.size(1) for image in images])
    max_width = max([image.size(2) for image in images])

    padded_images = []
    for image in images:
        # center image with padding
        padding_left = (max_width - image.shape[2]) // 2
        padding_right = max_width - image.shape[2] - padding_left
        padding_top = (max_height - image.shape[1]) // 2
        padding_bottom = max_height - image.shape[1] - padding_top

        padded_images.append(F.pad(image, (padding_left, padding_right, padding_top, padding_bottom)))

    targets = [item[1] for item in batch]

    padded_batch = list(zip(padded_images, targets))
    return padded_batch


def plot_loss_graph(train_losses=None, val_losses=None, test_losses=None, save_to="loss.png"):
    plt.clf()
    if train_losses is not None:
        x = np.arange(1, len(train_losses) + 1)
        plt.plot(x, train_losses, label="Train loss")
    if val_losses is not None:
        x = np.arange(1, len(val_losses) + 1)
        plt.plot(x, val_losses, label="Validation loss")
    if test_losses is not None:
        x = np.arange(1, len(test_losses) + 1)
        plt.plot(x, test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train loss and validation loss over epoch")
    plt.legend()
    plt.savefig(save_to)
