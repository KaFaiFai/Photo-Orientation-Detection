import torch
import matplotlib.pyplot as plt
import numpy as np


def variable_size_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # target = torch.LongTensor(target)
    # return [data, target]
    return batch


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
