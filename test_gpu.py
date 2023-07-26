import torch
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    a = torch.rand((70, 1000, 1000)).to(DEVICE)
    print("Finish initializing tensor a")
    print(torch.cuda.memory_summary())
    time.sleep(5)
    a = None
    print("Removing tensor a")
    print(torch.cuda.memory_summary())
    time.sleep(5)
    pass


if __name__ == "__main__":
    main()
