# Packages
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from skimage.io import imsave
from torchvision.utils import save_image
from utils import compute_gt_gradient, make_canvas_mask, numpy2tensor, laplacian_filter_tensor, \
                  MeanShift, Vgg16, gram_matrix
import argparse
import pdb
import os
import imageio.v2 as iio
import torch.nn.functional as F



parser = argparse.ArgumentParser()
parser.add_argument('--source_file', type=str, default='data/1_source.png', help='path to the source image')#源文件路径
parser.add_argument('--mask_file', type=str, default='data/1_mask.png', help='path to the mask image')#蒙版路径
parser.add_argument('--target_file', type=str, default='data/1_target.png', help='path to the target image')#目标文件路径
parser.add_argument('--output_dir', type=str, default='results/1', help='path to output')#输出路径
parser.add_argument('--ss', type=int, default=225, help='source image size')#源文件预设默认尺寸
parser.add_argument('--ts', type=int, default=512, help='target image size')#目标文件默认尺寸
parser.add_argument('--x', type=int, default=400, help='vertical location (center)')#200/235 blending位置的垂直坐标
parser.add_argument('--y', type=int, default=380, help='vertical location (center)')#blending位置的水平坐标
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')#训练使用的gpu
parser.add_argument('--num_steps', type=int, default=1000, help='Number of iterations in each pass')#每一个pass训练的epoches数
parser.add_argument('--save_video', type=bool, default=False, help='save the intermediate reconstruction process')#存储blending过程的视频 默认为false
opt = parser.parse_args()


os.makedirs(opt.output_dir, exist_ok = True)#生成输出的文件夹



###################################
########### First Pass ###########
###################################

# Inputs
source_file = opt.source_file
mask_file = opt.mask_file
target_file = opt.target_file

# Hyperparameter Inputs 超参数
gpu_id = opt.gpu_id
num_steps = opt.num_steps
ss = opt.ss; # source image size
ts = opt.ts # target image size
x_start = opt.x; y_start = opt.y # blending location

# Default weights for loss functions in the first pass 默认firstpass权重
grad_weight = 1e4; style_weight = 1e4; content_weight = 1; tv_weight = 1e-6

# Load Images 加载三张图片
source_img = np.array(Image.open(source_file).convert('RGB').resize((ss, ss)))
target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))
mask_img = np.array(Image.open(mask_file).convert('L').resize((ss, ss)))
mask_img[mask_img>0] = 1

# Make Canvas Mask
# Canvas蒙版：生成一个与目标图像尺寸一致的全0的numpy数组，并按照给定的xy，把mask图像的numpy数组填入全零的数组中的指定位置，
# 其次使用numpy2tensor函数把数组变换成张量，放入模型中计算。
# 最后使用.view()函数，把canvas_mask的size变为（3，ts，ts）；.repeat（）函数是为了尽可了能保留更多原始数据。
canvas_mask = make_canvas_mask(x_start, y_start, target_img, mask_img)
canvas_mask = numpy2tensor(canvas_mask, gpu_id)
canvas_mask = canvas_mask.squeeze(0).repeat(3,1).view(3,ts,ts).unsqueeze(0)

# Compute Ground-Truth Gradients 计算
gt_gradient = compute_gt_gradient(x_start, y_start, source_img, target_img, mask_img, gpu_id)

# Convert Numpy Images Into Tensors
source_img = torch.from_numpy(source_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
input_img = torch.randn(target_img.shape).to(gpu_id)

mask_img = numpy2tensor(mask_img, gpu_id)
mask_img = mask_img.squeeze(0).repeat(3,1).view(3,ss,ss).unsqueeze(0)

# Define LBFGS optimizer
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer
optimizer = get_input_optimizer(input_img)

# Define Loss Functions
mse = torch.nn.MSELoss()

# Import VGG network for computing style and content loss
mean_shift = MeanShift(gpu_id)
vgg = Vgg16().to(gpu_id)

# Save reconstruction process in a video
if opt.save_video:
    recon_process_video = iio.get_writer(os.path.join(opt.output_dir, 'recon_process.mp4'), format='FFMPEG', mode='I', fps=400)

run = [0]
while run[0] <= num_steps:
    
    def closure():
        # Composite Foreground and Background to Make Blended Image
        blend_img = torch.zeros(target_img.shape).to(gpu_id)
        blend_img = input_img*canvas_mask + target_img*(canvas_mask-1)*(-1) 
        
        # Compute Laplacian Gradient of Blended Image
        pred_gradient = laplacian_filter_tensor(blend_img, gpu_id)
        
        # Compute Gradient Loss
        grad_loss = 0
        for c in range(len(pred_gradient)):
            grad_loss += mse(pred_gradient[c], gt_gradient[c])
        grad_loss /= len(pred_gradient)
        grad_loss *= grad_weight
        
        # Compute Style Loss
        target_features_style = vgg(mean_shift(target_img))
        target_gram_style = [gram_matrix(y) for y in target_features_style]
        
        blend_features_style = vgg(mean_shift(input_img))
        blend_gram_style = [gram_matrix(y) for y in blend_features_style]
        
        style_loss = 0
        for layer in range(len(blend_gram_style)):
            style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
        style_loss /= len(blend_gram_style)  
        style_loss *= style_weight           

        
        # Compute Content Loss
        blend_obj = blend_img[:,:,int(x_start-source_img.shape[2]*0.5):int(x_start+source_img.shape[2]*0.5), int(y_start-source_img.shape[3]*0.5):int(y_start+source_img.shape[3]*0.5)]
        source_object_features = vgg(mean_shift(source_img*mask_img))
        blend_object_features = vgg(mean_shift(blend_obj*mask_img))
        content_loss = content_weight * mse(blend_object_features.relu2_2, source_object_features.relu2_2)
        content_loss *= content_weight
        
        # Compute TV Reg Loss
        tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
                   torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
        tv_loss *= tv_weight
        
        # Compute Total Loss and Update Image
        loss = grad_loss + style_loss + content_loss + tv_loss
        optimizer.zero_grad()
        loss.backward()

        # Write to output to a reconstruction video 
        if opt.save_video:
            foreground = input_img*canvas_mask
            foreground = (foreground - foreground.min()) / (foreground.max() - foreground.min())
            background = target_img*(canvas_mask-1)*(-1)
            background = background / 255.0
            final_blend_img =  + foreground + background
            if run[0] < 200:
                # more frames for early optimization by repeatedly appending the frames
                for _ in range(10):
                    recon_process_video.append_data(final_blend_img[0].transpose(0,2).transpose(0,1).cpu().data.numpy())
            else:
                recon_process_video.append_data(final_blend_img[0].transpose(0,2).transpose(0,1).cpu().data.numpy())
        
        # Print Loss
        if run[0] % 1 == 0:
            print("run {}:".format(run))
            print('grad : {:4f}, style : {:4f}, content: {:4f}, tv: {:4f}'.format(\
                          grad_loss.item(), \
                          style_loss.item(), \
                          content_loss.item(), \
                          tv_loss.item()
                          ))
            print()
        
        run[0] += 1
        return loss
    
    optimizer.step(closure)

# clamp the pixels range into 0 ~ 255
input_img.data.clamp_(0, 255)

# Make the Final Blended Image
blend_img = torch.zeros(target_img.shape).to(gpu_id)
blend_img = input_img*canvas_mask + target_img*(canvas_mask-1)*(-1) 
blend_img_np = blend_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]

# Save image from the first pass
first_pass_img_file = os.path.join(opt.output_dir, 'first_pass.png')
imsave(first_pass_img_file, blend_img_np.astype(np.uint8))

###################################
########### Second Pass ###########
###################################

# Default weights for loss functions in the second pass
style_weight = 1e7; content_weight = 1; tv_weight = 1e-6
ss = 512; ts = 512
num_steps = opt.num_steps

first_pass_img = np.array(Image.open(first_pass_img_file).convert('RGB').resize((ss, ss)))
target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))
first_pass_img = torch.from_numpy(first_pass_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)

first_pass_img = first_pass_img.contiguous()
target_img = target_img.contiguous()

# Define LBFGS optimizer
def get_input_optimizer(first_pass_img):
    optimizer = optim.LBFGS([first_pass_img.requires_grad_()])
    return optimizer

optimizer = get_input_optimizer(first_pass_img)

print('Optimizing...')
run = [0]
while run[0] <= num_steps:
    
    def closure():
        
        # Compute Loss Loss    
        target_features_style = vgg(mean_shift(target_img))
        target_gram_style = [gram_matrix(y) for y in target_features_style]
        blend_features_style = vgg(mean_shift(first_pass_img))
        blend_gram_style = [gram_matrix(y) for y in blend_features_style]
        style_loss = 0
        for layer in range(len(blend_gram_style)):
            style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
        style_loss /= len(blend_gram_style)  
        style_loss *= style_weight        
        
        # Compute Content Loss
        content_features = vgg(mean_shift(first_pass_img))
        content_loss = content_weight * mse(blend_features_style.relu2_2, content_features.relu2_2)
        
        # Compute Total Loss and Update Image
        loss = style_loss + content_loss
        optimizer.zero_grad()
        loss.backward()

        # Write to output to a reconstruction video 
        if opt.save_video:
            foreground = first_pass_img*canvas_mask
            foreground = (foreground - foreground.min()) / (foreground.max() - foreground.min())
            background = target_img*(canvas_mask-1)*(-1)
            background = background / 255.0
            final_blend_img =  + foreground + background
            recon_process_video.append_data(final_blend_img[0].transpose(0,2).transpose(0,1).cpu().data.numpy())
        
        # Print Loss
        if run[0] % 1 == 0:
            print("run {}:".format(run))
            print(' style : {:4f}, content: {:4f}'.format(\
                          style_loss.item(), \
                          content_loss.item()
                          ))
            print()
        
        run[0] += 1
        return loss
    
    optimizer.step(closure)

# clamp the pixels range into 0 ~ 255
first_pass_img.data.clamp_(0, 255)

# Make the Final Blended Image
input_img_np = first_pass_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]

# Save image from the second pass
imsave(os.path.join(opt.output_dir, 'second_pass.png'), input_img_np.astype(np.uint8))

# Save recon process video
if opt.save_video:
    recon_process_video.close()



