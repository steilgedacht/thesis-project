import torch

class Loss_BCE_Dice():
    """Combined BCEWithLogits + Dice loss.

    Parameters:
      pos_weight: optional scalar or tensor to pass to BCEWithLogitsLoss as
                  pos_weight to upweight positive examples (useful for class
                  imbalance). If provided, a BCEWithLogitsLoss with that
                  pos_weight is constructed. Otherwise the default
                  BCEWithLogitsLoss() is used.
    """
    def __init__(self, pos_weight=None, weight=None):
        # Create BCEWithLogitsLoss with optional pos_weight / weight
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float)
            try:
                # instantiate with pos_weight
                self.loss_fn_1 = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
            except Exception:
                # fallback to default
                self.loss_fn_1 = torch.nn.BCEWithLogitsLoss()
        elif weight is not None:
            w = torch.tensor(weight, dtype=torch.float)
            try:
                self.loss_fn_1 = torch.nn.BCEWithLogitsLoss(weight=w)
            except Exception:
                self.loss_fn_1 = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fn_1 = torch.nn.BCEWithLogitsLoss()

        self.loss_bce = 0.0
        self.loss_dice = 0.0

        self.report_losses()
    
    def __call__(self, pred, target, coords=None):
        # Ensure the BCE loss is computed on the same device as the tensors
        if isinstance(self.loss_fn_1, torch.nn.modules.loss._Loss):
            try:
                self.loss_fn_1 = self.loss_fn_1.to(pred.device)
            except Exception:
                pass

        self.loss_bce = self.loss_fn_1(pred, target)
        self.loss_dice = self.dice_loss(pred, target)

        self.report_losses()

        return self.loss_bce + self.loss_dice

    def report_losses(self):
        try:
            b = float(self.loss_bce.item())
        except Exception:
            try:
                b = float(self.loss_bce)
            except Exception:
                b = 0.0
        try:
            d = float(self.loss_dice.item())
        except Exception:
            try:
                d = float(self.loss_dice)
            except Exception:
                d = 0.0

        self.loss_to_report = {
            "training_loss_bce": b,
            "training_loss_dice": d,
        }

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

    def dice_score(self, pred, target, smooth=1e-6):
        with torch.no_grad():
            return 1 - self.dice_loss(pred, target, smooth)
