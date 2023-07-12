import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt


from dataset.Cityscapes import CityscapesDataset
from model.MobileNetV2 import MobileNetV2

# Hyperparameters etc.
load_dotenv()
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 1000
NUM_WORKERS = 2
IMAGE_SCALE = 0.1
LOAD_MODEL = False
DATA_ROOT = os.environ["CITYSCAPES_DATASET"]
EXP_FOLDER = "exp1"


def main():
    print("Init dataset ...")
    dataset_train = CityscapesDataset(DATA_ROOT, split="train", scale=IMAGE_SCALE)
    dataset_val = CityscapesDataset(DATA_ROOT, split="val", scale=IMAGE_SCALE)
    # subset to test if it overfits, comment this for full scale training
    dataset_train = Subset(dataset_train, np.arange(20))
    dataset_val = Subset(dataset_val, np.arange(10))
    ###
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

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
        # loop = tqdm(train_loader)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")

        model.train()
        cur_train_loss = 0
        for batch_idx, (image, label) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            # forward
            output = model(image)
            loss = criterion(output, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_train_loss += loss.item()
            if batch_idx % 10 == 0:
                print(
                    f"[Batch {batch_idx:4d}/{len(train_loader)}]"
                    f" Loss: {loss.item():.4f}"
                )
        cur_train_loss /= len(train_loader)
        train_losses.append(cur_train_loss)

        model.eval()
        cur_val_loss = 0
        for batch_idx, (image, label) in enumerate(val_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            # forward
            output = model(image)
            loss = criterion(output, label)

            cur_val_loss += loss.item()
        cur_val_loss /= len(val_loader)
        val_losses.append(cur_val_loss)
        print(f"Validation loss: {cur_val_loss:.4f}")

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

            image, label = dataset_train[0]
            image = image.to(DEVICE)
            output = model(image.unsqueeze(0)).squeeze().to("cpu")
            folder = Path("snapshot") / EXP_FOLDER / f"e{epoch:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            CityscapesDataset.plot_image(image, folder / "image.png")

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
