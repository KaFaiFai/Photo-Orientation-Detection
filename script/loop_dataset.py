"""
Loop through whole dataset, optionally optimize model and return loss
"""

import torch


def _loop_dataset(model,
                  dataloader,
                  criterion,
                  device,
                  optimizer=None,
                  silent=False):
    """
    used for training loop by setting optimizer or evaluation loop
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()

    cur_loss = 0
    for batch_idx, data in enumerate(dataloader):
        # feed forward with single image and accumulate gradients
        for image, label in data:
            # expect a single image (C, H, W) and integer label
            image = image.to(device).unsqueeze(0)
            label = torch.LongTensor([label]).to(device)

            # forward
            output = model(image)
            loss = criterion(output, label)

        # backward
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cur_loss += loss.item()
        if batch_idx % 10 == 0 and not silent:
            print(f"[Batch {batch_idx:4d}/{len(dataloader)}]"
                  f" Loss: {loss.item():.4f}")
    cur_loss /= len(dataloader)
    return cur_loss


def train_loop(model, dataloader, criterion, device, optimizer):
    return _loop_dataset(model, dataloader, criterion, device, optimizer)


def eval_loop(model, dataloader, criterion, device):
    return _loop_dataset(model, dataloader, criterion, device, silent=True)
