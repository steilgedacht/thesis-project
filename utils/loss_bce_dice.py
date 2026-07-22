import torch

class Loss_BCE_Dice():
    def __init__(self, loss_fn_1=torch.nn.BCEWithLogitsLoss()):
        self.loss_fn_1 = loss_fn_1
        self.loss_bce = 0
        self.loss_dice = 0
    
    def __call__(self, pred, target):
        self.loss_bce = self.loss_fn_1(pred, target)
        self.loss_dice = self.dice_loss(pred, target)
        return self.loss_bce + self.loss_dice

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))
