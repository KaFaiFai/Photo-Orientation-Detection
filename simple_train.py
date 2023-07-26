import timeit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv

from dataset import CityscapesDataset, ImagenetDataset
from model.MobileNetV2 import MobileNetV2
from model.EfficientNet import EfficientNet
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
IMAGE_SCALE = 0.5
LOAD_FROM = None
DATASET = CityscapesDataset
DATA_ROOT = os.environ["CITYSCAPES_DATASET"]
EXP_FOLDER = "exp1"


def main():
    print("Init dataset ...")
    dataset_train = DATASET(DATA_ROOT, split="train", scale=IMAGE_SCALE)
    dataset_val = DATASET(DATA_ROOT, split="val", scale=IMAGE_SCALE)
    # subset to test if it overfits, comment this for full scale training
    # dataset_train = Subset(dataset_train, np.arange(200))
    # dataset_val = Subset(dataset_val, np.arange(20))
    ###

    identity_collate = lambda batch: batch
    train_loader = DataLoader(
        dataset_train, BATCH_SIZE, shuffle=False, collate_fn=identity_collate
    )
    val_loader = DataLoader(
        dataset_val, BATCH_SIZE, shuffle=False, collate_fn=identity_collate
    )

    print(f"Init model using {DEVICE=} ...")
    model = MobileNetV2(4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction="sum")  # to get average easily
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if LOAD_FROM is not None:
        model.load_state_dict(torch.load(LOAD_FROM))

    train_losses = []
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")

        model.train()
        start_time = timeit.default_timer()
        cur_train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            batch_size = len(data)
            total_image_size = 0
            # feed forward with single image and accumulate gradients
            for image, label in data:
                # expect a single image (C, H, W) and integer label
                image = image.to(DEVICE)
                label = torch.LongTensor([label]).to(DEVICE)

                output = model(image.unsqueeze(0))
                loss = criterion(output, label)

                total_image_size += image.shape[1] * image.shape[2]

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_train_loss += loss.item()
            if batch_idx % 1 == 0:
                # if every image is a square, their equivalent side will be
                avg_image_side = (total_image_size / batch_size) ** 0.5
                print(
                    f"[Batch {batch_idx:4d}/{len(train_loader)}]"
                    f"| Loss: {loss.item()/batch_size:.4f}"
                    f"| Memory: {torch.cuda.memory_reserved(0)/1024**3:.4f}GB"
                    f"| Image size: {int(avg_image_side)}"
                )

        cur_train_loss /= len(train_loader)
        train_losses.append(cur_train_loss)
        # train_metrics = ClassificationMetrics(train_truths, train_outputs)

        end_time = timeit.default_timer()
        train_time = end_time - start_time
        print(
            f"Train time: {train_time:.2f}s,"
            f" {train_time/len(train_loader):.2f}s/batch"
        )

        print("")


if __name__ == "__main__":
    main()
