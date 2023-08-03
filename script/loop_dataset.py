"""
Loop through whole dataset, optionally optimize model and return loss
"""

import torch
import timeit
import numpy as np


def _loop_dataset(
    model,
    dataloader,
    criterion,
    device,
    optimizer=None,
    silent=False,
    return_samples=True,
    max_batch=None,
):
    """
    used for training loop by setting optimizer or evaluation loop
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()

    start_time = timeit.default_timer()

    total_loss = 0
    all_truths = []
    all_outputs = []

    samples = ([], [], [])
    for batch_idx, data in enumerate(dataloader):
        if max_batch is not None and batch_idx > max_batch:
            break

        batch_size = len(data)

        # feed forward with multiple images of same size
        image = [d[0] for d in data]
        image = torch.stack(image).to(device)
        label = [d[1] for d in data]
        label = torch.LongTensor(label).to(device)

        output = model(image)
        loss = criterion(output, label)

        # save records
        all_truths += label.detach().cpu().tolist()
        all_outputs.append(output.detach().cpu())
        if return_samples and batch_idx == 0:
            samples = (image.detach().cpu(), label.detach().cpu(), output.detach().cpu())

        # backward
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if batch_idx % 5 == 0 and not silent:
            print(
                f"[Batch {batch_idx:4d}/{len(dataloader)}]"
                f" Loss: {loss.item()/batch_size:.4f}"
                f"| Memory: {torch.cuda.memory_reserved(0)/1024**3:.4f}GB"
            )

    total_loss /= len(dataloader)
    all_outputs = torch.cat(all_outputs)  # from list of tensor to numpy array
    all_outputs = np.array(all_outputs).squeeze()

    end_time = timeit.default_timer()
    time_spent = end_time - start_time

    if return_samples:
        return total_loss, all_truths, all_outputs, time_spent, samples
    return total_loss, all_truths, all_outputs, time_spent


def train_loop(model, dataloader, criterion, device, optimizer):
    return _loop_dataset(model, dataloader, criterion, device, optimizer, max_batch=1000)


def eval_loop(model, dataloader, criterion, device):
    return _loop_dataset(model, dataloader, criterion, device, silent=True, max_batch=200)
