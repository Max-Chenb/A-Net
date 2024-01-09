import torch
import torch.nn as nn
import torch.nn.functional as F

class DWconv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation = 1, norm_layer=nn.BatchNorm2d):
        super(DWconv, self).__init__()
        self.dconv = nn.Conv2d(in_channels,in_channels,kernel_size=3,dilation=dilation,padding=dilation,groups=in_channels,bias=False)
        self.pconv = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)

    def forward(self,x):
        x = self.dconv(x)
        x = self.pconv(x)
        return x

class Bottleneck(nn.Module):

    def __init__(self, in_channels, inter_channels, out_channels, cat_channels = 0, wide=False, concat_=False,
                 downsampling=False, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.downsamping = downsampling
        self.concat_ = concat_

        if concat_:
            self.concat_conv = DWconv(in_channels+cat_channels,in_channels)

        if downsampling:
            self.maxpool = nn.MaxPool2d(2, 2)
            self.conv_down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels)
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            norm_layer(inter_channels),
            nn.PReLU()
        )

        if downsampling:
            self.conv2 = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, 2, stride=2, bias=False),
                norm_layer(inter_channels),
                nn.PReLU()
            )

        else:
            if wide:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(inter_channels, inter_channels, (5, 1), padding=(2, 0), bias=False),
                    nn.Conv2d(inter_channels, inter_channels, (1, 5), padding=(0, 2), bias=False),
                    # nn.Conv2d(inter_channels, inter_channels, kernel_size=5, padding=2, bias=False),
                    norm_layer(inter_channels),
                    nn.PReLU()
                )

            else:
                self.conv2 = nn.Sequential(
                    DWconv(inter_channels, inter_channels),
                    norm_layer(inter_channels),
                    nn.PReLU()
                )

        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.Dropout2d(0.1)
        )
        self.act = nn.PReLU()

    def forward(self, x):
        if self.concat_:
            x = self.concat_conv(x)

        identity = x
        if self.downsamping:
            identity = self.maxpool(identity)
            identity = self.conv_down(identity)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act(out + identity)

        return out

class UpsamplingBottleneck(nn.Module):
    """upsampling Block"""

    def __init__(self, in_channels, inter_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(UpsamplingBottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.upsampling = nn.MaxUnpool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.PReLU(),
            nn.ConvTranspose2d(inter_channels, inter_channels, 2, 2, bias=False),
            norm_layer(inter_channels),
            nn.PReLU(),
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.Dropout2d(0.1)
        )
        self.act = nn.PReLU()

    def forward(self, x, max_indices = None):
        out_up = self.conv(x)
        if max_indices == None:
            out_up = self.up(out_up)
        else:
            out_up = self.upsampling(out_up, max_indices)

        out_ext = self.block(x)
        out = self.act(out_up + out_ext)
        return out

class SegHead(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super(SegHead, self).__init__()
        self.fullconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, inter_channels, 2, 2, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(inter_channels)
        )
        self.conv1 = nn.Sequential(
            DWconv(inter_channels,inter_channels),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU()
        )

        self.conv_out = nn.Conv2d(inter_channels,out_channels,kernel_size=3,padding=1)

    def forward(self,x):
        out = self.fullconv(x)
        out = self.conv1(out)
        out = self.conv_out(out)
        return out

class ANet(nn.Module):
    def __init__(self, nclass = 1):
        super(ANet, self).__init__()
        #downsample
        self.bottleneck0_0 = Bottleneck(3, 8, 32, downsampling=True)
        self.bottleneck0_1 = Bottleneck(32, 8, 32)
        self.bottleneck0_2 = Bottleneck(32, 8, 32, wide=True)

        self.bottleneck1_0 = Bottleneck(32, 16, 64, downsampling=True)
        self.bottleneck1_1 = Bottleneck(64, 16, 64)
        self.bottleneck1_2 = Bottleneck(64, 16, 64, wide=True)

        self.bottleneck2_0 = Bottleneck(64, 32, 128, downsampling=True)
        self.bottleneck2_1 = Bottleneck(128, 32, 128)
        self.bottleneck2_2 = Bottleneck(128, 32, 128, wide=True)

        self.bottleneck3_0 = Bottleneck(128, 32, 128, downsampling=True)
        self.bottleneck3_1 = Bottleneck(128, 32, 128)
        self.bottleneck3_2 = Bottleneck(128, 32, 128, wide=True)
        self.bottleneck3_3 = Bottleneck(128, 32, 128)
        self.bottleneck3_4 = Bottleneck(128, 32, 128, wide=True)

        self.bottleneck4_0 = Bottleneck(128, 32, 128, downsampling=True)
        self.bottleneck4_1 = Bottleneck(128, 32, 128)
        self.bottleneck4_2 = Bottleneck(128, 32, 128, wide=True)
        self.bottleneck4_3 = Bottleneck(128, 32, 128)
        self.bottleneck4_4 = Bottleneck(128, 32, 128, wide=True)

        #upsample
        self.bottleneck5_0 = UpsamplingBottleneck(128, 32, 128)
        self.bottleneck5_1 = Bottleneck(128, 32, 128)
        self.bottleneck5_2 = Bottleneck(128, 32, 128, wide=True)

        self.bottleneck6_0 = UpsamplingBottleneck(128, 32, 128)
        self.bottleneck6_1 = Bottleneck(128, 32, 128)
        self.bottleneck6_2 = Bottleneck(128, 32, 128, wide=True)

        self.bottleneck7_0 = UpsamplingBottleneck(128, 16, 64)
        self.bottleneck7_1 = Bottleneck(64, 16, 64, cat_channels=64, concat_=True)
        self.bottleneck7_2 = Bottleneck(64, 16, 64, wide=True)

        self.bottleneck8_0 = UpsamplingBottleneck(64, 8, 32)
        self.bottleneck8_1 = Bottleneck(32, 8, 32)
        self.bottleneck8_2 = Bottleneck(32, 8, 32, wide=True)

        self.seg = SegHead(in_channels=32,inter_channels=16,out_channels=nclass)

    def forward(self, x):
        x_shape = x.shape
        # init
        x = self.bottleneck0_0(x)
        x = self.bottleneck0_1(x)
        x0 = self.bottleneck0_2(x)

        # stage 1
        x = self.bottleneck1_0(x0)
        x = self.bottleneck1_1(x)
        x1 = self.bottleneck1_2(x)

        # stage 2
        x = self.bottleneck2_0(x1)
        x = self.bottleneck2_1(x)
        x2 = self.bottleneck2_2(x)

        # stage 3
        x3 = self.bottleneck3_0(x2)
        x3 = self.bottleneck3_1(x3)
        x3 = self.bottleneck3_2(x3)
        x3 = self.bottleneck3_3(x3)
        x3 = self.bottleneck3_4(x3)

        # stage 4
        x4 = self.bottleneck4_0(x3)
        x4 = self.bottleneck4_1(x4)
        x4 = self.bottleneck4_2(x4)
        x4 = self.bottleneck4_3(x4)
        x4 = self.bottleneck4_4(x4)

        # stage 5
        x = self.bottleneck5_0(x4)
        x = self.bottleneck5_1(x)
        x = self.bottleneck5_2(x)

        x = self.bottleneck6_0(x)
        x = self.bottleneck6_1(x)
        x = self.bottleneck6_2(x)

        x = self.bottleneck7_0(x)
        x = self.bottleneck7_1(torch.cat([x1,x],dim=1))
        x = self.bottleneck7_2(x)

        x = self.bottleneck8_0(x)
        x = self.bottleneck8_1(x)
        x = self.bottleneck8_2(x)

        # out
        # x = self.fullconv(x)
        x = self.seg(x)
        return x

if __name__ == '__main__':
    img = torch.randn(1, 3, 512, 512)
    model = ANet(nclass=1)
    output = model(img)
    print(output.size())
