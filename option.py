import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=100000)
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument('--resume',type=bool,default=True)
parser.add_argument('--eval_step',type=int,default=5000)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir',type=str,default='./trained_models/')
parser.add_argument('--trainset',type=str,default='rw2ah_train')
parser.add_argument('--testset',type=str,default='rw2ah_test')
parser.add_argument('--net',type=str,default='ffa')
parser.add_argument('--gps',type=int,default=3,help='residual_groups')
parser.add_argument('--blocks',type=int,default=20,help='residual_blocks')
parser.add_argument('--bs',type=int,default=16,help='batch size')
parser.add_argument('--crop',action='store_true')
parser.add_argument('--crop_size',type=int,default=240,help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche',action='store_true',help='no lr cos schedule')
parser.add_argument('--perloss',action='store_true',help='perceptual loss')#action表示指定这个参数为true，没有false
# 添加SSIM损失参数（可选是否使用SSIM损失，以及SSIM损失的类型：全局或局部或不使用）
parser.add_argument('--ssim_loss', action='store_true', help='use ssim loss')
parser.add_argument('--ssim_loss_type', type=str, default='none', choices=['none', 'global', 'region'], help='ssim loss type: none, global, or region')
# 可选L1Loss，sky or not
parser.add_argument('--skyl1', action='store_true', help='use skyl1 loss')
# 添加LPIPS参数
parser.add_argument('--lpips_eval', action='store_true', help='evaluate using LPIPS')
parser.add_argument('--lpips_net', type=str, default='vgg', choices=['alex', 'vgg', 'squeeze'], help='LPIPS network type')
# 添加多尺度损失参数(可选各个损失权重)
parser.add_argument('--multi_scale_loss', action='store_true', help='use multi-scale loss')
parser.add_argument('--ms_l1_weight', type=float, default=1.0, help='weight for L1 loss in multi-scale loss')
parser.add_argument('--ms_ssim_weight', type=float, default=0.5, help='weight for SSIM loss in multi-scale loss')
parser.add_argument('--ms_fft_weight', type=float, default=0.1, help='weight for FFT loss in multi-scale loss')

opt=parser.parse_args()

opt.device='cuda' if torch.cuda.is_available() else 'cpu'
model_name=opt.trainset+'_'+opt.net.split('.')[0]+'_'+str(opt.gps)+'_'+str(opt.blocks)
opt.model_dir=opt.model_dir+model_name+'.pk'
log_dir='logs/'+model_name

print(opt)
print('model_dir:',opt.model_dir)


if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('numpy_files'):
	os.mkdir('numpy_files')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('samples'):
	os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
	os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
	os.mkdir(log_dir)
