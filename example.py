import torch
import torch.nn as nn
import torch.nn.functional as F

#reference : https://github.com/joe-siyuan-qiao/WeightStandardization

class Conv3d(nn.Conv3d):
    def __init__(self, in_channels, output_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv3d, self).__init__(in_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        w = self.weight
        w_mean = w.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        w = w - w_mean
        std = w.view(w.size(0), -1).std(dim=1).view(-1,1,1,1,1) + 1e-5
        w = w / std.expand_as(w)
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


if __name__ == '__main__':
    conv3d = Conv3d(in_channels=3, output_channels=8, kernel_size=1)
    # b, c, z, h, w
    x = torch.randn(8, 3, 5, 32, 32).float()
    x = conv3d(x)

    print(x)
    
