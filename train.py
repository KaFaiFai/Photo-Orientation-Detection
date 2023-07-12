import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import timeit

from dataset.Cityscapes import CityscapesDataset
from model.MobileNetV2 import MobileNetV2
from script.loop_dataset import train_loop, eval_loop

# Hyperparameters etc.
load_dotenv()
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 1000
NUM_WORKERS = 2
IMAGE_SCALE = 1
LOAD_MODEL = False
DATA_ROOT = os.environ["CITYSCAPES_DATASET"]
EXP_FOLDER = "exp1"


def main():
    print("Init dataset ...")
    dataset_train = CityscapesDataset(DATA_ROOT,
                                      split="train",
                                      scale=IMAGE_SCALE)
    dataset_val = CityscapesDataset(DATA_ROOT, split="val", scale=IMAGE_SCALE)
    # subset to test if it overfits, comment this for full scale training
    dataset_train = Subset(dataset_train, np.arange(20))
    dataset_val = Subset(dataset_val, np.arange(10))
    ###

    identity_collate = lambda batch: batch
    train_loader = DataLoader(dataset_train,
                              BATCH_SIZE,
                              shuffle=False,
                              collate_fn=identity_collate)
    val_loader = DataLoader(dataset_val,
                            BATCH_SIZE,
                            shuffle=False,
                            collate_fn=identity_collate)

    print(f"Init model using {DEVICE=} ...")
    model = MobileNetV2(4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction="sum")  # to get average easily
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # if LOAD_MODEL:
    #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # check_accuracy(val_loader, model, device=DEVICE)

    train_losses = []
    val_losses = []
    for epoch in range(NUM_EPOCHS):
        start_time = timeit.default_timer()
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")

        cur_train_loss = train_loop(model, train_loader, criterion, DEVICE,
                                    optimizer)
        train_losses.append(cur_train_loss)

        cur_val_loss = eval_loop(model, val_loader, criterion, DEVICE)
        val_losses.append(cur_val_loss)
        print(f"Validation loss: {cur_val_loss:.4f}")

        end_time = timeit.default_timer()
        print(f"Time: {end_time-start_time:.2f}s")
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # save_checkpoint(checkpoint)

        # # check accuracy
        # check_accuracy(val_loader, model, device=DEVICE)

        # save model, some examples and graphs to a folder
        if epoch % 20 == 0:
            print("save snapshot")

            image, label = dataset_val[0]
            image = image.to(DEVICE)
            output = model(image.unsqueeze(0)).squeeze().to("cpu")
            prediction = torch.argmax(output).item()
            folder = Path("snapshot") / EXP_FOLDER / f"e{epoch:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            CityscapesDataset.plot_image(image, save_to=folder / "image.png")
            CityscapesDataset.plot_image(image,
                                         num_rotation=4 - prediction,
                                         save_to=folder / "output.png")

            x = np.arange(1, epoch + 2)
            plt.clf()
            plt.plot(x, train_losses, label="Train loss")
            plt.plot(x, val_losses, label="Validation loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Train loss and validation loss over epoch")
            plt.legend()
            plt.savefig(folder / "loss.png")

        print("")


if __name__ == "__main__":
    main()
