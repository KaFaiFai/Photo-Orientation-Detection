"""
Loop through whole dataset, optionally optimize model and return loss
"""

import torch

def _loop_dataset(model,
                  dataloader,
                  criterion,
                  device,
                  optimizer=None,
                  silent=False,
                  return_samples=True,
                  image_same_size=False):
    """
    used for training loop by setting optimizer or evaluation loop
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()

    cur_loss = 0
    samples = ([], [], [])
    for batch_idx, data in enumerate(dataloader):
        if image_same_size:
            # feed forward with multiple images of same size
            image = [torch.Tensor(d[0]) for d in data]
            image = torch.stack(image).to(device)
            label = [d[1] for d in data]
            label = torch.LongTensor(label).to(device)

            output = model(image)
            loss = criterion(output, label)

            # save a few samples
            if return_samples and batch_idx == 0:
                samples = (image, label, output)
        else:
            # feed forward with single image and accumulate gradients
            for image, label in data:
                # expect a single image (C, H, W) and integer label
                image = image.to(device)
                label = torch.LongTensor([label]).to(device)

                output = model(image.unsqueeze(0))
                loss = criterion(output, label)

                # save a few samples
                if return_samples and batch_idx == 0:
                    samples[0].append(image)
                    samples[1].append(label)
                    samples[2].append(output)

        # backward
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cur_loss += loss.item()
        if batch_idx % 50 == 0 and not silent:
            print(f"[Batch {batch_idx:4d}/{len(dataloader)}]"
                  f" Loss: {loss.item():.4f}")
    cur_loss /= len(dataloader)

    if return_samples:
        return cur_loss, samples
    return cur_loss


def train_loop(model, dataloader, criterion, device, optimizer):
    return _loop_dataset(model, dataloader, criterion, device, optimizer)


def eval_loop(model, dataloader, criterion, device):
    return _loop_dataset(model, dataloader, criterion, device, silent=True)
