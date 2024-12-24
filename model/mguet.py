# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 16, (128, 128))
        self.down1 = down(16, 32, (64, 64))
        self.down2 = down(32, 64, (32, 32))
        self.down3 = down(64, 128, (16, 16))
        self.down4 = down(128, 128, (8, 8))
        self.up1 = up(256, 64, (16, 16), True)
        self.up2 = up(128, 32, (32, 32), True)
        self.up3 = up(64, 16, (64, 64), True)
        self.up4 = up(32, 16, (128, 128),  True)
        self.outc = outconv(16, n_classes)
        # self.conv1 = nn.Conv2d(64, 32, 9, padding=4)
        # self.conv2 = nn.Conv2d(32, 32, 1, padding=0)
        # self.conv3 = nn.Conv2d(32, n_classes, 5, padding=2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)
        out_put = self.outc(x_up4)
        # x = torch.softmax(x, dim=1)
        # x = torch.argmax(x, dim=1, keepdim=True)
        # x = self.conv1(x)
        # x = nn.ReLU(inplace=True)(x)
        # x = self.conv2(x)
        # x = nn.ReLU(inplace=True)(x)
        # x = self.conv3(x)
        return out_put


class CoordGate(nn.Module):
    def __init__(self, in_channels, out_channels, img_size):
        super(CoordGate, self).__init__()
        self.img_size = img_size

        # 使用 1x1 卷积代替全连接层进行坐标编码
        self.coord_conv1 = nn.Conv2d(2, 32, kernel_size=1)
        self.coord_conv2 = nn.Conv2d(32, 256, kernel_size=1)
        self.coord_conv3 = nn.Conv2d(256, out_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        coord_x = torch.linspace(-1, 1, self.img_size[0]).view(1, -1).expand(self.img_size[1], self.img_size[0])
        coord_y = torch.linspace(-1, 1, self.img_size[1]).view(-1, 1).expand(self.img_size[1], self.img_size[0])
        coord = torch.stack([coord_x, coord_y], dim=0).unsqueeze(0).to(x.device)
        gate = self.relu(self.coord_conv1(coord))
        gate = self.relu(self.coord_conv2(gate))
        gate = self.coord_conv3(gate)

        gate = gate.expand(x.size(0), -1, -1, -1)  
        conv_out = self.conv(x) # 卷积
        out = conv_out * gate   # 门控信号

        return out


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, img_size):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            CoordGate(out_ch, out_ch, img_size)
            # nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, imgsize):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, imgsize)
        # self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        y = self.conv(x)
        return y


class down(nn.Module):
    def __init__(self, in_ch, out_ch, img_size):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, img_size)
        )

    def forward(self, x):
        y = self.mpconv(x)
        return y


class up(nn.Module):
    def __init__(self, in_ch, out_ch, imgsize, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, img_size=imgsize)

    def forward(self, x1, x2):
        x1_1 = self.up(x1)

        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x3 = torch.cat([x2, x1_1], dim=1)
        y = self.conv(x3)
        return y


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        # self.conv = F.conv2d()

    def forward(self, x):
        y = self.conv(x)
        return y


if __name__ == '__main__':
    inputs = torch.randn(4, 1, 128, 128)
    net = UNet(1, 1)
    print(net)
    out = net(inputs)
    print(out)
    print(out.shape)
    CUDA = torch.cuda.is_available
    if CUDA:
        model = net.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    input = torch.randn(4, 1, 128, 128).cuda()
    flops, params = profile(model.cuda(), inputs=(input,))
    print(" %.4f | %.4f" % (params / (1000 ** 2), flops / (1000 ** 3)))  # 这里除以1000的平方，是为了化成M的单位，
