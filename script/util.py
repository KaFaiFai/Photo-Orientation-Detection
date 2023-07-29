import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def variable_size_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # target = torch.LongTensor(target)
    # return [data, target]
    return batch


def crop_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    max_height = max([img.size(1) for img in image_batch])
    max_width = max([img.size(2) for img in image_batch])

    image_batch = [
        # The needed padding is the difference between the
        # max width/height and the image's actual width/height.
        F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])
        for img in image_batch
    ]
    # return [data, target]


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
