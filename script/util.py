import torch


def variable_size_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # target = torch.IntTensor(target)
    # return [data, target]
    return batch
