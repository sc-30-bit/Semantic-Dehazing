import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from pytorch_msssim import SSIM  # Import SSIM from pytorch_msssim
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
from torchvision.models import vgg16
print('log_dir :',log_dir)
print('model_name:',model_name)

models_={
	'ffa':FFA(gps=opt.gps,blocks=opt.blocks),
}
loaders_={
	'rw2ah_train':RW2AH_train_loader,
	'rw2ah_test':RW2AH_test_loader,
}
start_time=time.time()
T=opt.steps	
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net,loader_train,loader_test,optim,criterion):
	losses=[]
	start_step=0
	max_ssim=0
	max_psnr=0
	min_lpips = float('inf') if opt.lpips_eval else None  # 初始化LPIPS的最佳值（越低越好）
	ssims=[]
	psnrs=[]
	lpips_list = [] if opt.lpips_eval else None
	if opt.resume and os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		ckp=torch.load(opt.model_dir)
		losses=ckp['losses']
		net.load_state_dict(ckp['model'])
		start_step=ckp['step']
		max_ssim=ckp['max_ssim']
		max_psnr=ckp['max_psnr']
		psnrs=ckp['psnrs']
		ssims=ckp['ssims']
		print(f'start_step:{start_step} start training ---')
	else :
		print('train from scratch *** ')
	# Initialize SSIM loss (data_range=1.0 for normalized [0, 1] images)
	ssim_loss_fn = SSIM(data_range=1.0, size_average=False, channel=3).to(opt.device)
	# Persisten writer for the whole training
	writer = SummaryWriter(log_dir=log_dir, comment=model_name)
	train_iter = iter(loader_train)
	for step in range(start_step+1,opt.steps+1):
		net.train()
		lr=opt.lr
		if not opt.no_lr_sche:
			lr=lr_schedule_cosdecay(step,T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr  
		# Get batch from RW2AH_Dataset: (hazy_imgs, clear_imgs, weight_mask, binary_mask)
		# x, y, weight_mask, binary_mask = next(iter(loader_train))
		try:
			x, y, weight_mask, binary_mask = next(train_iter)
		except StopIteration:
			train_iter = iter(loader_train)
			x, y, weight_mask, binary_mask = next(train_iter)
		x = x.to(opt.device)  # Hazy images [batch, 3, H, W]
		y = y.to(opt.device)  # Clear images [batch, 3, H, W]
		weight_mask = weight_mask.to(opt.device)  # Weight mask [batch, 1, H, W], values [1.0, 1.5]
		binary_mask = binary_mask.to(opt.device)  # Binary mask [batch, 1, H, W], values [0, 1]
		out=net(x)
		#three channels output RGB
		if opt.multi_scale_loss:
			# 计算多尺度损失
			loss = criterion[1](out, y)
		else:
			# 计算单尺度损失
			loss=criterion[0](out,y)* weight_mask
			#loss=criterion[0](out,y)
			loss=loss.mean()  # Average loss over the batch
			ssim_loss = 0
		
			if opt.ssim_loss and opt.ssim_loss_type == 'global':
				# 计算全局 SSIM 损失
				ssim_val = ssim_loss_fn(out, y)
				ssim_loss = 1 - ssim_val.mean()
				loss = loss + 0.5 * ssim_loss
			elif opt.ssim_loss and opt.ssim_loss_type == 'region':
				# 仅对掩码区域计算SSIM
				masked_out = out * binary_mask
				masked_y = y * binary_mask
				# 计算SSIM损失（注意：SSIM值越高表示图像越相似，所以取1-SSIM作为损失）
				ssim_val = ssim_loss_fn(masked_out, masked_y)
				ssim_loss = 1 - ssim_val.mean()
				loss = loss + 0.5 * ssim_loss
		#optional(perloss)
		if opt.perloss:
			loss2=criterion[2](out,y)
			loss=loss+0.04*loss2
		
		loss.backward()

		'''if step == start_step+1:
			# 归一化weight_mask到[0, 1]范围
			weight_mask_normalized = weight_mask / 1.5
			tensorShow([x.cpu(), y.cpu(), out.cpu(), weight_mask_normalized.cpu(), binary_mask.cpu()],
              ['Hazy', 'Clear', 'Pred', 'Weight Mask', 'Binary Mask'])'''
		
		optim.step()
		optim.zero_grad()
		losses.append(loss.item())
		print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

		writer.add_scalar('data/loss',loss,step)

		if step % opt.eval_step ==0 and step > 0:
			with torch.no_grad():
				if opt.lpips_eval:
					val_ssim, val_psnr, val_lpips = test(net, loader_test, max_psnr, max_ssim, step)
					print(f'\nstep :{step} |val_psnr :{val_psnr:.4f}|val_ssim:{val_ssim:.4f}|val_lpips:{val_lpips:.4f}')
				else:
					val_ssim, val_psnr = test(net, loader_test, max_psnr, max_ssim, step)
					print(f'\nstep :{step} |val_psnr :{val_psnr:.4f}|val_ssim:{val_ssim:.4f}')
			
			# 统一通过同一个 writer 记录评估结果
			writer.add_scalar('data/ssim',val_ssim,step)
			writer.add_scalar('data/psnr',val_psnr,step)
			# 创建动态group字典
			group_dict = {
                'ssim': val_ssim,
                'psnr': val_psnr,
                'loss': loss
            }
			# 添加LPIPS标量记录,条件添加lpips到group
			if opt.lpips_eval:
				writer.add_scalar('data/lpips', val_lpips, step)
				group_dict['lpips'] = val_lpips
			writer.add_scalars('group', group_dict, step)

			ssims.append(val_ssim)
			psnrs.append(val_psnr)
			if opt.lpips_eval:
				lpips_list.append(val_lpips)
				if val_ssim > max_ssim and val_psnr > max_psnr and val_lpips < min_lpips:
					max_ssim=max(max_ssim,val_ssim)
					max_psnr=max(max_psnr,val_psnr)
					min_lpips = min(min_lpips, val_lpips)
					torch.save({
								'step':step,
								'max_psnr':max_psnr,
								'max_ssim':max_ssim,
								'min_lpips':min_lpips,
								'ssims':ssims,
								'psnrs':psnrs,
								'lpips_list':lpips_list,
								'losses':losses,
								'model':net.state_dict(),
						},opt.model_dir)
					print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')
			elif val_ssim > max_ssim and val_psnr > max_psnr :
				max_ssim=max(max_ssim,val_ssim)
				max_psnr=max(max_psnr,val_psnr)
				torch.save({
							'step':step,
							'max_psnr':max_psnr,
							'max_ssim':max_ssim,
							'ssims':ssims,
							'psnrs':psnrs,
							'losses':losses,
							'model':net.state_dict()
				},opt.model_dir)
				print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')
			print(f'\nstep :{step} |max_psnr :{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

	np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy',losses)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy',ssims)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy',psnrs)
	if opt.lpips_eval:
		np.save(f'./numpy_files/{model_name}_{opt.steps}_lpips.npy', lpips_list)
	# 关闭writer
	writer.close()

def test(net,loader_test,max_psnr,max_ssim,step):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	lpips_values = [] if opt.lpips_eval else None  # 初始化LPIPS列表
	for i , batch in enumerate(loader_test):
		inputs=batch[0].to(opt.device)
		targets=batch[1].to(opt.device)
		pred=net(inputs)
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
		if opt.lpips_eval:
			lpips_val = calculate_lpips(pred, targets, net_type=opt.lpips_net, device=opt.device)
			lpips_values.append(lpips_val)
	# 返回结果
	if opt.lpips_eval:
		return np.mean(ssims), np.mean(psnrs), np.mean(lpips_values)
	else:
		return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
	loader_train=loaders_[opt.trainset]
	loader_test=loaders_[opt.testset]
	net=models_[opt.net]
	net=net.to(opt.device)
	if opt.device=='cuda':
		net=torch.nn.DataParallel(net)
		cudnn.benchmark=True
	criterion = []
	criterion.append(nn.L1Loss(reduction='none').to(opt.device))
	# 多尺度损失
	if opt.multi_scale_loss:
		criterion.append(MultiScaleLoss(l1_weight=opt.ms_l1_weight, ssim_weight=opt.ms_ssim_weight, fft_weight=opt.ms_fft_weight).to(opt.device))
	if opt.perloss:
			vgg_model = vgg16(pretrained=True).features[:16]
			vgg_model = vgg_model.to(opt.device)
			for param in vgg_model.parameters():
				param.requires_grad = False
			criterion.append(PerLoss(vgg_model).to(opt.device))
	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	train(net,loader_train,loader_test,optimizer,criterion)
	

