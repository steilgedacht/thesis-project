import torch

class Loss_BCE_Dice():
    """Combined BCEWithLogits + Dice loss + optional TV regularization.

    Parameters:
      use_dynamic_pos_weight: if True, compute pos_weight from the class imbalance
                              in each batch's ground truth (default True for adaptive weighting).
                              If False, uses a fixed pos_weight value.
      pos_weight: fixed scalar pos_weight (only used if use_dynamic_pos_weight=False).
                  For lesion prediction, typically 2-10.
      weight: alternative weighting for BCE (not commonly used with pos_weight).
      use_tv_loss: if True, add total variation regularization for spatial smoothness.
      tv_weight: coefficient for TV loss term (default 0.01).
    """
    def __init__(self, use_dynamic_pos_weight=True, 
                 use_tv_loss=False, tv_weight=0.01, edge_weight=0.0):
        self.use_dynamic_pos_weight = use_dynamic_pos_weight
        self.edge_weight = edge_weight
        
        # Create base BCEWithLogitsLoss without pos_weight (we'll apply it dynamically)
        self.loss_fn_1 = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.use_tv_loss = use_tv_loss
        self.tv_weight = tv_weight
        self.loss_bce = 0.0
        self.loss_dice = 0.0
        self.loss_tv = 0.0
        self.current_pos_weight = 1.0  # Track the applied weight for logging

        self.report_losses()
    
    def __call__(self, pred, target):
        # Compute dynamic pos_weight from class imbalance if enabled
        if self.use_dynamic_pos_weight:
            positive_pixels = target.sum().item()
            total_pixels = target.numel()
            negative_pixels = total_pixels - positive_pixels
            
            # pos_weight = ratio of negative to positive pixels
            # Avoids division by zero with small epsilon
            self.current_pos_weight = max(1.0, negative_pixels / (positive_pixels + 1e-7))
            
            # Create BCEWithLogitsLoss with computed pos_weight for this batch
            pw = torch.tensor(self.current_pos_weight, dtype=torch.float, device=pred.device)
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred, target, pos_weight=pw, reduction='none'
            )
        else:
            # Use default BCE without pos_weight
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred, target, reduction='none'
            )

        if self.edge_weight > 0:
            edge_band = self._edge_band(target)
            bce_loss = bce_loss * (1.0 + self.edge_weight * edge_band)
        self.loss_bce = bce_loss.mean()
        
        self.loss_dice = self.dice_loss(pred, target)
        
        total_loss = self.loss_bce + self.loss_dice
        
        if self.use_tv_loss:
            self.loss_tv = self.tv_loss(pred)
            total_loss = total_loss + self.tv_weight * self.loss_tv
        else:
            self.loss_tv = 0.0

        self.report_losses()

        return total_loss

    @staticmethod
    def _edge_band(target):
        """Return a one-voxel boundary band for 3D binary targets."""
        if target.ndim != 5:
            raise ValueError(f"Expected [N, C, D, H, W] target, got {target.shape}")
        dilated = torch.nn.functional.max_pool3d(target, kernel_size=3, stride=1, padding=1)
        eroded = -torch.nn.functional.max_pool3d(-target, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

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
        try:
            tv = float(self.loss_tv.item()) if isinstance(self.loss_tv, torch.Tensor) else float(self.loss_tv)
        except Exception:
            tv = 0.0

        self.loss_to_report = {
            "training_loss_bce": b,
            "training_loss_dice": d,
        }
        
        # Log the current pos_weight for monitoring class imbalance
        if self.use_dynamic_pos_weight:
            self.loss_to_report["dynamic_pos_weight"] = float(self.current_pos_weight)
        
        if self.use_tv_loss:
            self.loss_to_report["training_loss_tv"] = tv

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

    def dice_score(self, pred, target, smooth=1e-6):
        with torch.no_grad():
            return 1 - self.dice_loss(pred, target, smooth)

    def tv_loss(self, pred):
        """
        Total Variation loss for spatial smoothness.
        Encourages neighboring voxels to have similar predicted values,
        reducing spurious noise and promoting coherent lesion regions.
        
        Computed on the predicted logits (not probabilities).
        """
        # Compute differences along spatial dimensions (D, H, W)
        # pred shape: [B, T, 1, D, H, W] for LSTM or [B, 1, D, H, W] for single step
        
        # Handle both 6D (sequences) and 5D (single frame) tensors
        if pred.dim() == 6:
            # Batch, Time, Channels, D, H, W
            diff_d = torch.abs(pred[:, :, :, 1:, :, :] - pred[:, :, :, :-1, :, :]).mean()
            diff_h = torch.abs(pred[:, :, :, :, 1:, :] - pred[:, :, :, :, :-1, :]).mean()
            diff_w = torch.abs(pred[:, :, :, :, :, 1:] - pred[:, :, :, :, :, :-1]).mean()
        else:
            # Batch, Channels, D, H, W (single step)
            diff_d = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]).mean()
            diff_h = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]).mean()
            diff_w = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]).mean()
        
        return diff_d + diff_h + diff_w

