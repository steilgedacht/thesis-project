import torch


class Loss_BCE_Dice_TV:
    def __init__(
        self,
        lambda_space=1e-4,
        lambda_time=1e-3,
        pos_weight=1.0,
        device=None,
        use_dynamic_pos_weight=False,
    ):
        self.pos_weight = pos_weight
        self.use_dynamic_pos_weight = use_dynamic_pos_weight
        self.current_pos_weight = pos_weight
        self.lambda_space = lambda_space
        self.lambda_time = lambda_time

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_bce_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

        self.loss_bce = 0.0
        self.loss_dice = 0.0
        self.loss_tv_space = 0.0
        self.loss_tv_time = 0.0
        self.report_losses()

    def __call__(self, pred, target, coords=None):
        if self.use_dynamic_pos_weight:
            positive_pixels = target.sum().item()
            negative_pixels = target.numel() - positive_pixels
            self.current_pos_weight = max(
                1.0, negative_pixels / (positive_pixels + 1e-7)
            )
        else:
            self.current_pos_weight = self.pos_weight

        self.loss_bce_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                self.current_pos_weight,
                dtype=pred.dtype,
                device=pred.device,
            )
        )
        self.loss_bce = self.loss_bce_fn(pred, target)
        self.loss_dice = self.dice_loss(
            pred, target, pos_weight=self.current_pos_weight
        )

        total_loss = self.loss_bce + self.loss_dice

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
            "training_loss_bce": self._loss_value(self.loss_bce),
            "training_loss_dice": self._loss_value(self.loss_dice),
            "training_loss_tv_space": self._loss_value(self.loss_tv_space),
            "training_loss_tv_time": self._loss_value(self.loss_tv_time),
            "dynamic_pos_weight": float(self.current_pos_weight),
        }

    @staticmethod
    def _loss_value(value):
        if isinstance(value, float):
            return value
        return value.detach().cpu().item()

    def dice_loss(self, pred, target, pos_weight=1.0, smooth=1e-6):
        pred = torch.sigmoid(pred)
        weights = torch.where(target == 1, pos_weight, 1.0)
        intersection = (pred * target * weights).sum()
        denominator = (pred * weights).sum() + (target * weights).sum()
        return 1 - (2.0 * intersection + smooth) / (denominator + smooth)

    def compute_tv_losses(self, pred, coords):
        probabilities = torch.sigmoid(pred)
        gradients = torch.autograd.grad(
            outputs=probabilities,
            inputs=coords,
            grad_outputs=torch.ones_like(probabilities),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        tv_space = torch.abs(gradients[..., :3]).sum(dim=-1).mean()
        tv_time = torch.abs(gradients[..., 3:]).mean()
        return tv_space, tv_time

    def dice_score(self, pred, target, smooth=1e-6):
        with torch.no_grad():
            return 1 - self.dice_loss(pred, target, smooth=smooth)
