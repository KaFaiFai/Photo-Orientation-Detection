import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv

from dataset.Cityscapes import CityscapesDataset
from model.MobileNetV2 import MobileNetV2
from script.loop_dataset import train_loop, eval_loop
from script.util import plot_loss_graph
from script.metrics import ClassificationMetrics

# Hyperparameters etc.
load_dotenv()
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 1000
NUM_WORKERS = 2
IMAGE_SCALE = 0.1
LOAD_MODEL = False
DATA_ROOT = os.environ["CITYSCAPES_DATASET"]
EXP_FOLDER = "exp3"


def main():
    print("Init dataset ...")
    dataset_train = CityscapesDataset(DATA_ROOT, split="train", scale=IMAGE_SCALE)
    dataset_val = CityscapesDataset(DATA_ROOT, split="val", scale=IMAGE_SCALE)
    # subset to test if it overfits, comment this for full scale training
    dataset_train = Subset(dataset_train, np.arange(400))
    dataset_val = Subset(dataset_val, np.arange(40))
    ###

    identity_collate = lambda batch: batch
    train_loader = DataLoader(dataset_train, BATCH_SIZE, shuffle=True, collate_fn=identity_collate)
    val_loader = DataLoader(dataset_val, BATCH_SIZE, shuffle=True, collate_fn=identity_collate)

    print(f"Init model using {DEVICE=} ...")
    model = MobileNetV2(4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction="sum")  # to get average easily
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # if LOAD_MODEL:
    #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    train_losses = []
    val_losses = []
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")

        train_info = train_loop(model, train_loader, criterion, DEVICE, optimizer)
        train_loss, train_truths, train_outputs, train_time, train_samples = train_info
        train_losses.append(train_loss)
        train_metrics = ClassificationMetrics(train_truths, train_outputs)
        print(f"Train time: {train_time:.2f}s,"
              f" {train_time/len(train_loader):.2f}s/batch")

        val_info = eval_loop(model, val_loader, criterion, DEVICE)
        val_loss, val_truths, val_outputs, val_time, val_samples = val_info
        val_losses.append(val_loss)
        val_metrics = ClassificationMetrics(val_truths, val_outputs)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        print("--- Validation report ---")
        val_metrics.print_report()

        del train_info, train_loss, train_truths, train_outputs
        del val_info, val_loss, val_truths, val_outputs

        # save model
        # checkpoint = {
        #     "state_dict": model.state_dict(),
        #     "optimizer": optimizer.state_dict(),
        # }
        # save_checkpoint(checkpoint)

        # save model, some examples and graphs to a folder
        if epoch % 5 == 0:
            print("saving snapshot")
            epoch_folder = Path("snapshot") / EXP_FOLDER / f"e{epoch:03d}"
            epoch_folder.mkdir(parents=True, exist_ok=True)

            images, truths, outputs = val_samples
            sample_metrics = ClassificationMetrics(truths, outputs)
            predictions = sample_metrics.preds.tolist()
            CityscapesDataset.plot_results(images, predictions, save_to=epoch_folder / "output.png")
            del images, outputs, predictions

            plot_loss_graph(train_losses, val_losses, save_to=epoch_folder / "loss.png")
            del train_losses, val_losses

        del train_samples, val_samples
        print("")


if __name__ == "__main__":
    main()
