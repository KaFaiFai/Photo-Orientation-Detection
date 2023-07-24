import torch
from torch import nn

def EfficientNet(num_class):
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    model.classifier[3] = nn.Linear(1280, num_class)
    return model

def test():
    from torchinfo import summary

    batch_size = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = EfficientNet().to(device)
    summary(net, input_size=(batch_size, 3, 523, 523))

    # verify it works for variable input size
    y = net(torch.randn((batch_size, 3, 523, 523)).to(device))
    y = net(torch.randn((batch_size, 3, 329, 175)).to(device))


    # network_state = net.state_dict()
    # print("PyTorch model's state_dict:")
    # for layer, tensor in network_state.items():
    #     print(f"{layer:<45}: {tensor.size()}")


if __name__ == "__main__":
    test()