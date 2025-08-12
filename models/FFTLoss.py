import torch
import torch.nn as nn


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)


class MultiScaleLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.5, fft_weight=0.1, reduction='mean'):
        super(MultiScaleLoss, self).__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.fft_weight = fft_weight
        self.l1_criterion = nn.L1Loss(reduction=reduction)
        self.fft_criterion = FFTLoss(loss_weight=1.0, reduction=reduction)
        from pytorch_msssim import SSIM
        self.ssim_criterion = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, pred, target, weight_mask=None):
        # 多尺度损失计算
        loss = 0
        for scale in [(1.0, 1.0), (0.5, 0.5), (0.25, 0.25)]:
            scale_factor, weight = scale
            if scale_factor < 1.0:
                # 调整图像大小
                pred_scaled = nn.functional.interpolate(pred, scale_factor=scale_factor, mode='bilinear', align_corners=False)
                target_scaled = nn.functional.interpolate(target, scale_factor=scale_factor, mode='bilinear', align_corners=False)
                if weight_mask is not None:
                    weight_mask_scaled = nn.functional.interpolate(weight_mask, scale_factor=scale_factor, mode='bilinear', align_corners=False)
                else:
                    weight_mask_scaled = None
            else:
                pred_scaled = pred
                target_scaled = target
                weight_mask_scaled = weight_mask

            # 计算L1损失
            l1_loss = self.l1_criterion(pred_scaled, target_scaled)
            if weight_mask_scaled is not None:
                l1_loss = l1_loss * weight_mask_scaled
                l1_loss = l1_loss.mean()

            # 计算SSIM损失
            ssim_val = self.ssim_criterion(pred_scaled, target_scaled)
            ssim_loss = (1 - ssim_val)

            # 计算FFT损失
            fft_loss = self.fft_criterion(pred_scaled, target_scaled)

            # 加权求和
            scale_loss = weight * (self.l1_weight * l1_loss + self.ssim_weight * ssim_loss + self.fft_weight * fft_loss)
            loss += scale_loss

        return loss