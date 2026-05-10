"""
Custom U-Net.

Implementation from:
https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
"""

from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        init_features=32,
        kernel_size=3,
        sigmoid=True,
    ):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(
            in_channels, features, name="enc1", kernel_size=kernel_size
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(
            features, features * 2, name="enc2", kernel_size=kernel_size
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(
            features * 2, features * 4, name="enc3", kernel_size=kernel_size
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(
            features * 4, features * 8, name="enc4", kernel_size=kernel_size
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(
            features * 8, features * 16, name="bottleneck", kernel_size=kernel_size
        )

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block(
            (features * 8) * 2, features * 8, name="dec4", kernel_size=kernel_size
        )
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block(
            (features * 4) * 2, features * 4, name="dec3", kernel_size=kernel_size
        )
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block(
            (features * 2) * 2, features * 2, name="dec2", kernel_size=kernel_size
        )
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(
            features * 2, features, name="dec1", kernel_size=kernel_size
        )

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = sigmoid

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1)) if self.sigmoid else self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name, kernel_size=3):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def inflate_3x3_to_5x5(w):
    w5 = torch.zeros(
        w.shape[0],
        w.shape[1],
        5,
        5,
        dtype=w.dtype,
        device=w.device,
    )
    w5[:, :, 1:4, 1:4] = w
    return w5


def load_lightning_3x3_into_unet_5x5(model_5x5, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    old_state = ckpt["state_dict"]

    new_model_state = model_5x5.state_dict()
    converted_state = {}

    loaded_direct = 0
    inflated = 0
    skipped = []

    for key, value in old_state.items():
        # Lightning checkpoint has "model." prefix
        if key.startswith("model."):
            key = key[len("model.") :]

        if key not in new_model_state:
            skipped.append((key, "not in new model"))
            continue

        target = new_model_state[key]

        if value.shape == target.shape:
            converted_state[key] = value
            loaded_direct += 1

        elif (
            value.ndim == 4
            and target.ndim == 4
            and value.shape[:2] == target.shape[:2]
            and value.shape[-2:] == (3, 3)
            and target.shape[-2:] == (5, 5)
        ):
            converted_state[key] = inflate_3x3_to_5x5(value)
            inflated += 1

        else:
            skipped.append((key, f"{tuple(value.shape)} -> {tuple(target.shape)}"))

    missing, unexpected = model_5x5.load_state_dict(converted_state, strict=False)

    print(f"Loaded directly: {loaded_direct}")
    print(f"Inflated 3x3 -> 5x5: {inflated}")
    print(f"Skipped: {len(skipped)}")
    print(f"Missing: {len(missing)}")
    print(f"Unexpected: {len(unexpected)}")

    if skipped:
        print("\nSkipped examples:")
        for item in skipped[:20]:
            print(item)

    return model_5x5
