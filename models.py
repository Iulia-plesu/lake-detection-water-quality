import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=2):
        super().__init__()

        self.c1 = conv_block(in_channels, 64)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = conv_block(64, 128)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = conv_block(128, 256)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = conv_block(256, 512)
        self.p4 = nn.MaxPool2d(2)

        self.c5 = conv_block(512, 1024)

        self.u6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.c6 = conv_block(1024, 512)
        self.u7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c7 = conv_block(512, 256)
        self.u8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c8 = conv_block(256, 128)
        self.u9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c9 = conv_block(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.p1(c1))
        c3 = self.c3(self.p2(c2))
        c4 = self.c4(self.p3(c3))
        c5 = self.c5(self.p4(c4))

        x = self.c6(torch.cat([self.u6(c5), c4], dim=1))
        x = self.c7(torch.cat([self.u7(x), c3], dim=1))
        x = self.c8(torch.cat([self.u8(x), c2], dim=1))
        x = self.c9(torch.cat([self.u9(x), c1], dim=1))

        return self.out(x)


class MultiSpectralDeepLabV3(nn.Module):
    def __init__(self, num_bands=6, num_classes=2):
        super(MultiSpectralDeepLabV3, self).__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

        if num_bands != 3:
            old_conv = self.model.backbone.conv1
            self.model.backbone.conv1 = nn.Conv2d(
                num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                new_weight = torch.zeros(64, num_bands, 7, 7)
                new_weight[:, :3] = old_conv.weight
                avg_w = old_conv.weight.mean(dim=1, keepdim=True)
                for i in range(3, num_bands):
                    new_weight[:, i:i+1] = avg_w
                self.model.backbone.conv1.weight = nn.Parameter(new_weight)

        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']