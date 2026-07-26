import torch

class Loss_BCE_Dice_TV:
    def __init__(
        self,
        loss_fn_1=torch.nn.BCEWithLogitsLoss(),
        lambda_space=1e-4,
        lambda_time=1e-3,
    ):
        self.loss_fn_1 = loss_fn_1
        self.lambda_space = lambda_space
        self.lambda_time = lambda_time

        self.loss_bce = 0.0
        self.loss_dice = 0.0
        self.loss_tv_space = 0.0
        self.loss_tv_time = 0.0

        self.report_losses()


    def __call__(self, pred, target, coords=None):
        self.loss_bce = self.loss_fn_1(pred, target)
        self.loss_dice = self.dice_loss(pred, target)

        total_loss = self.loss_bce + self.loss_dice

        # Total Variation 
        if coords is not None and (
            self.lambda_space > 0 or self.lambda_time > 0
        ):
            tv_space, tv_time = self.compute_tv_losses(pred, coords)

            self.loss_tv_space = tv_space * self.lambda_space
            self.loss_tv_time = tv_time * self.lambda_time

            total_loss += self.loss_tv_space + self.loss_tv_time

        self.report_losses()

        return total_loss

    def report_losses(self):
        self.loss_to_report = {
            "training_loss_bce": self.loss_bce if isinstance(self.loss_bce, float) else self.loss_bce.cpu().detach().item(),
            "training_loss_dice": self.loss_dice if isinstance(self.loss_dice, float) else self.loss_dice.cpu().detach().item(),
            "training_loss_tv_space": self.loss_tv_space if isinstance(self.loss_tv_space, float) else self.loss_tv_space.cpu().detach().item(),
            "training_loss_tv_time": self.loss_tv_time if isinstance(self.loss_tv_time, float) else self.loss_tv_time.cpu().detach().item(),
        }

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (
            (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        )

    def compute_tv_losses(self, pred, coords):
        """Berechnet die räumliche (Space) und zeitliche (Time) Ableitung der Predictions

        bezüglich der Koordinaten (x, y, z, t).
        """
        # logits to probabilities
        prob_pred = torch.sigmoid(pred)
        # Autograd-Gradienten von pred bzgl. coords berechnen
        # Output-Shape von gradients: [Batch, N_Points, 4] -> (dx, dy, dz, dt)
        gradients = torch.autograd.grad(
            outputs=prob_pred,
            inputs=coords,
            grad_outputs=torch.ones_like(prob_pred),
            create_graph=True,  # Essenziell für den Backpropagation-Schritt des Total Loss!
            retain_graph=True,
            only_inputs=True,
        )[0]

        # spatial component (x, y, z) -> indices 0, 1, 2
        grad_spatial = gradients[..., :3]
        tv_space = torch.abs(grad_spatial).sum(dim=-1).mean()

        # temporal component (t) -> index 3
        grad_time = gradients[..., 3:]
        tv_time = torch.abs(grad_time).mean()

        return tv_space, tv_time