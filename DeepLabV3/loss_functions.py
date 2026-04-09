import torch
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):

        preds = torch.sigmoid(preds)

        smooth = 1

        intersection = (
            preds * targets
        ).sum()

        union = preds.sum() + targets.sum()

        dice = (
            2 * intersection + smooth
        ) / (union + smooth)

        return 1 - dice
    
class BCEDiceLoss(nn.Module):

    def __init__(self):
        super(BCEDiceLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):

        bce_loss = self.bce(
            preds,
            targets
        )

        dice_loss = self.dice(
            preds,
            targets
        )

        return bce_loss + dice_loss
    
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.8, gamma=2):

        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):

        bce = nn.functional.binary_cross_entropy_with_logits(
            preds,
            targets,
            reduction='none'
        )

        pt = torch.exp(-bce)

        focal = self.alpha * (
            (1 - pt) ** self.gamma
        ) * bce

        return focal.mean()