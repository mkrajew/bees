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
