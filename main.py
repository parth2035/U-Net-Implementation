import torch
from model.UNet import UNet

torch.manual_seed(0)


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used: {}".format(DEVICE))

    # UNet
    u_net = UNet(in_channels=1, out_channels=2, hidden_channels=64).to(DEVICE)
    print(u_net)

    random_test = torch.randn((1, 1, 572, 572)).to(DEVICE)
    out = u_net(random_test)
    print("Output is of size: {}".format(out.size()))

    del u_net

    print("Program has Ended")