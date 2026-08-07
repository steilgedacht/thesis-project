import torch

class Loss_BCE_Dice():
    def __init__(self, loss_fn_1=torch.nn.BCEWithLogitsLoss()):
        self.loss_fn_1 = loss_fn_1
        self.loss_bce = 0
        self.loss_dice = 0

        self.report_losses()
    
    def __call__(self, pred, target, coords=None):
        self.loss_bce = self.loss_fn_1(pred, target)
        self.loss_dice = self.dice_loss(pred, target)

        self.report_losses()

        return self.loss_bce + self.loss_dice

    def report_losses(self):
        self.loss_to_report = {
            "training_loss_bce": self.loss_bce,
            "training_loss_dice": self.loss_dice,
        }

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

    def dice_score(self, pred, target, smooth=1e-6):
        with torch.no_grad():
            return 1 - self.dice_loss(pred, target, smooth)
