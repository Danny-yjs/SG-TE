import torch
from torch import nn


class Adapt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=1)
        self.last_conv = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.kblock = nn.Sequential(nn.BatchNorm1d(4),
                                    nn.ReLU(),
                                    self.conv2)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        b = x1.shape[0]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        y = torch.cat((x1, x2), dim=1)
        y_c = self.conv1(y)
        y_black = self.kblock(y_c)
        y_black = self.kblock(y_black)
        y_out = self.conv3(y_black)
        y_s = y + y_out
        w = self.soft(y_s)
        w1 = w[:, 0, :].unsqueeze(1)
        w2 = w[:, 1, :].unsqueeze(1)
        x1 = x1*w1
        x2 = x2*w2
        out = x1+x2
        out = out.view(b, -1)

        return out


# if __name__ == '__main__':
#     a = torch.rand(4, 256)
#     b = torch.rand(4, 256)
#     model = Adapt()
#     out = model.forward(a, b)
#     print(out.shape)
