import torch
from torch import nn

class Deblur(nn.Module):
    def __init__(self):
        super(Deblur, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    a = torch.randn(1, 3, 256, 256)
    deblur = Deblur()
    o = deblur(a)
    print(o.size())