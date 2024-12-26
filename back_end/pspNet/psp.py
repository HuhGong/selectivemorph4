import torch
from torch import nn
from torch.nn import functional as F


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.relu(x)
        return output


class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super().__init__()

        in_channel, out_channel, kernel_size, stride, padding, dilation, bias = 3, 64, 3, 2, 1, 1, False
        self.cbnr_1 = conv2DBatchNormRelu(in_channel, out_channel, kernel_size, stride,
                                          padding, dilation, bias)
        in_channel, out_channel, kernel_size, stride, padding, dilation, bias = 64, 64, 3, 1, 1, 1, False
        self.cbnr_2 = conv2DBatchNormRelu(in_channel, out_channel, kernel_size, stride,
                                          padding, dilation, bias)

        in_channel, out_channel, kernel_size, stride, padding, dilation, bias = 64, 128, 3, 1, 1, 1, False
        self.cbnr_3 = conv2DBatchNormRelu(in_channel, out_channel, kernel_size, stride,
                                          padding, dilation, bias)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        output = self.maxpool(x)
        return output


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return x


class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super().__init__()
        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation,
                                         dilation=dilation, bias=False)
        self.cb_3 = conv2DBatchNorm(mid_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                    dilation=1, bias=False)

        self.cb_residual = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)
        return self.relu(conv + residual)


class bottleNeckIdentityPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super().__init__()
        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation,
                                         dilation=dilation, bias=False)
        self.cb_3 = conv2DBatchNorm(mid_channels, in_channels, kernel_size=1, stride=1, padding=0,
                                    dilation=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = x
        return self.relu(conv + residual)


class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super().__init__()
        self.add_module("block1",
                        bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))
        for i in range(n_blocks - 1):
            self.add_module("block" + str(i + 2),
                            bottleNeckIdentityPSP(out_channels, mid_channels, stride, dilation))


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super().__init__()

        self.height, self.width = height, width
        out_channels = int(in_channels / len(pool_sizes))

        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                         bias=False)

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, bias=False)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, bias=False)

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                         dilation=1, bias=False)

    def forward(self, x):
        out1 = self.cbr_1(self.avpool_1(x))
        out1 = F.interpolate(out1, size=(self.height, self.width), mode="bilinear", align_corners=True)
        out2 = self.cbr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=(self.height, self.width), mode="bilinear", align_corners=True)
        out3 = self.cbr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=(self.height, self.width), mode="bilinear", align_corners=True)
        out4 = self.cbr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=(self.height, self.width), mode="bilinear", align_corners=True)

        output = torch.cat([x, out1, out2, out3, out4], dim=1)
        return output


class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super().__init__()

        self.height, self.width = height, width

        self.cbr = conv2DBatchNormRelu(in_channel=4096,
                                       out_channel=512,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=512,
                                        out_channels=n_classes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), mode="bilinear", align_corners=True
        )
        return output


class AuxiliaryPSPlayers(nn.Module):
    def __init__(self, in_channels,
                 height,
                 wdith,
                 n_classes):
        super().__init__()
        self.width, self.height = wdith, height

        self.cbr = conv2DBatchNormRelu(in_channel=in_channels,
                                       out_channel=256,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=256,
                                        out_channels=n_classes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), mode="bilinear", align_corners=True
        )
        return output


class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        block_config = [3, 4, 6, 3]
        img_size = 475
        img_size_8 = 60

        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(n_blocks=block_config[0],
                                              in_channels=128,
                                              mid_channels=64,
                                              out_channels=256,
                                              stride=1,
                                              dilation=1)
        self.feature_res_2 = ResidualBlockPSP(n_blocks=block_config[1],
                                              in_channels=256,
                                              mid_channels=128,
                                              out_channels=512,
                                              stride=2,
                                              dilation=1)
        self.feature_dilated_res_1 = ResidualBlockPSP(
            n_blocks=block_config[2],
            in_channels=512,
            mid_channels=256,
            out_channels=1024,
            stride=1, dilation=2
        )
        self.feature_dilated_res_2 = ResidualBlockPSP(
            n_blocks=block_config[3],
            in_channels=1024,
            mid_channels=512,
            out_channels=2048,
            stride=1, dilation=4
        )
        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[6, 3, 2, 1], height=img_size_8,
                                              width=img_size_8)
        self.decode_feature = DecodePSPFeature(height=img_size,
                                               width=img_size,
                                               n_classes=n_classes)
        self.aux = AuxiliaryPSPlayers(in_channels=1024,
                                      height=img_size,
                                      wdith=img_size,
                                      n_classes=n_classes)

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)

        output_aux = self.aux(x)

        x = self.feature_dilated_res_2(x)
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)
        return output, output_aux

