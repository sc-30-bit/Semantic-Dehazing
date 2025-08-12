from math import exp
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from  torchvision.transforms import ToPILImage
import lpips  # 添加LPIPS库导入

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1=torch.clamp(img1,min=0,max=1)
    img2=torch.clamp(img2,min=0,max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
def psnr(pred, gt):
    pred=pred.clamp(0,1).cpu().numpy()
    gt=gt.clamp(0,1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10( 1.0 / rmse)
def calculate_lpips(pred, gt, net_type='vgg', device=None):
    """
    计算LPIPS指标
    pred: 预测图像 [batch, 3, H, W]
    gt: 真实图像 [batch, 3, H, W]
    net_type: 特征提取网络类型，可选 'alex', 'vgg', 'squeeze'
    device: 设备
    """
    # 确保图像在[0,1]范围内
    pred = torch.clamp(pred, min=0, max=1)
    gt = torch.clamp(gt, min=0, max=1)

    # 初始化LPIPS模型
    lpips_model = lpips.LPIPS(net=net_type)
    if device is not None:
        lpips_model = lpips_model.to(device)

    # 计算LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(pred, gt)

    return lpips_value.mean().item()

if __name__ == "__main__":
    pass
