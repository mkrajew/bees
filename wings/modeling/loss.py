"""
Loss functions for the landmark masks.

`DiceLoss` is the Dice loss of Mateusz Buda (brain-segmentation-pytorch):
https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py
Only the number formatting of the class differs. `WeightedDiceLoss` and `IoULoss` are variations of it (they
reuse its input handling); `BCEDiceLoss` is independent.

The original code is distributed under the MIT License, whose copyright and permission notice are
reproduced below. The rest belongs to WingAI (GPLv3, see LICENSE in the repository root).

    MIT License

    Copyright (c) 2019 mateuszbuda

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1.0 - dsc


class WeightedDiceLoss(nn.Module):
    def __init__(self, landmark_weight=50.0, background_weight=1.0, smooth=1.0):
        super().__init__()
        self.landmark_weight = landmark_weight
        self.background_weight = background_weight
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()

        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)

        # waga 50 dla landmarków, 1 dla tła
        weights = torch.where(
            y_true > 0.5,
            torch.tensor(
                self.landmark_weight, device=y_true.device, dtype=y_true.dtype
            ),
            torch.tensor(
                self.background_weight, device=y_true.device, dtype=y_true.dtype
            ),
        )

        intersection = (weights * y_pred * y_true).sum()

        dsc = (2.0 * intersection + self.smooth) / (
            (weights * y_pred).sum() + (weights * y_true).sum() + self.smooth
        )

        return 1.0 - dsc


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)

        intersection = (y_pred * y_true).sum()
        total = y_pred.sum() + y_true.sum()
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=50.0, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        targets = targets.float()

        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)

        smooth = 1.0
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth))
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
