# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:28:28 2019

@author: Owen and Tarmily
"""

from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch
from torchvision import models
from collections import namedtuple
import time

import asyncio


def numpy2tensor(np_array, gpu_id):#ndarray转换为tensor张量
    if len(np_array.shape) == 2:
        tensor = torch.from_numpy(np_array).unsqueeze(0).float().to(gpu_id)
        #2d输入在第0维增加一个维度作为批处理大小1的张量
    else:
        tensor = torch.from_numpy(np_array).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
        #同理，但是交换1，3、2，3维度 输入是h，w，c，
        # 需要改为张量的默认形式n，c，h，w
    return tensor


def make_canvas_mask(x_start, y_start, target_img, mask):#构造画布掩码（类似蒙版）白色为1，可以操作图像内容；黑色为0，不可操作图像内容
    canvas_mask = np.zeros((target_img.shape[0], target_img.shape[1]))
    canvas_mask[int(x_start-mask.shape[0]*0.5):int(x_start+mask.shape[0]*0.5), int(y_start-mask.shape[1]*0.5):int(y_start+mask.shape[1]*0.5)] = mask
    return canvas_mask

def laplacian_filter_tensor(img_tensor, gpu_id):#拉普拉斯滤波器

    laplacian_filter = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
    #这个滤波器可以检测到图像中的水平和垂直边缘，-1表示相邻像素权重（水平以及垂直），4表示当前像素权重
    laplacian_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    laplacian_conv.weight = nn.Parameter(torch.from_numpy(laplacian_filter).float().unsqueeze(0).unsqueeze(0).to(gpu_id))
    #填入初始化的拉普拉斯卷积算子(二维矩阵升维满足卷积的4维需求height,width,channel,outputchannel)
    
    for param in laplacian_conv.parameters():
        param.requires_grad = False
        #这个函数不需要计算滤波器的梯度，因为滤波器是预定义的，可以提高效率
    
    red_img_tensor = img_tensor[:,0,:,:].unsqueeze(1)
    green_img_tensor = img_tensor[:,1,:,:].unsqueeze(1)
    blue_img_tensor = img_tensor[:,2,:,:].unsqueeze(1)
    #这个函数使用PyTorch的索引操作符来实现这一点。
    # 这个函数使用[:,0,:,:]来选择所有批次、第0、1、2个通道、所有高度和所有宽度的元素。
    # 然后，它使用unsqueeze(1)来在第1维上添加一个维度，以便将其转换为4D张量
    
    red_gradient_tensor = laplacian_conv(red_img_tensor).squeeze(1)
    #这个函数需要将滤波器应用于输入的图像张量的每个通道，squeeze（1）移除了第一维即通道数，变为3d张量
    green_gradient_tensor = laplacian_conv(green_img_tensor).squeeze(1) 
    blue_gradient_tensor = laplacian_conv(blue_img_tensor).squeeze(1)
    return red_gradient_tensor, green_gradient_tensor, blue_gradient_tensor
    

def compute_gt_gradient(x_start, y_start, source_img, target_img, mask, gpu_id):
    
    # compute source image gradient
    source_img_tensor = torch.from_numpy(source_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)#转换成tensor标准形式，同上
    red_source_gradient_tensor, green_source_gradient_tensor, blue_source_gradient_tenosr = laplacian_filter_tensor(source_img_tensor, gpu_id)
    #调用拉普拉斯滤波器，得到红绿蓝三色的梯度（图像中的水平和垂直边缘）
    red_source_gradient = red_source_gradient_tensor.cpu().data.numpy()[0]
    green_source_gradient = green_source_gradient_tensor.cpu().data.numpy()[0]
    blue_source_gradient = blue_source_gradient_tenosr.cpu().data.numpy()[0]
    
    # compute target image gradient
    target_img_tensor = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
    red_target_gradient_tensor, green_target_gradient_tensor, blue_target_gradient_tenosr = laplacian_filter_tensor(target_img_tensor, gpu_id)#目标图像的各个channel的梯度
    red_target_gradient = red_target_gradient_tensor.cpu().data.numpy()[0]
    #使用data.numpy()来将张量转换为numpy数组，使用[0]来选择第0个批次的元素
    green_target_gradient = green_target_gradient_tensor.cpu().data.numpy()[0]
    blue_target_gradient = blue_target_gradient_tenosr.cpu().data.numpy()[0]    
    
    # mask and canvas mask 构造出了与目标图像尺寸相同的新蒙版
    canvas_mask = np.zeros((target_img.shape[0], target_img.shape[1]))
    canvas_mask[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = mask
    
    # foreground gradient
    red_source_gradient = red_source_gradient * mask#源图像的红色通道的梯度
    green_source_gradient = green_source_gradient * mask
    blue_source_gradient = blue_source_gradient * mask
    red_foreground_gradient = np.zeros((canvas_mask.shape))
    red_foreground_gradient[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = red_source_gradient
    #把红色通道的梯度放入前景梯度的指定位置中（blending位置）
    green_foreground_gradient = np.zeros((canvas_mask.shape))
    green_foreground_gradient[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = green_source_gradient
    blue_foreground_gradient = np.zeros((canvas_mask.shape))
    blue_foreground_gradient[int(x_start-source_img.shape[0]*0.5):int(x_start+source_img.shape[0]*0.5), int(y_start-source_img.shape[1]*0.5):int(y_start+source_img.shape[1]*0.5)] = blue_source_gradient
    
    # background gradient
    red_background_gradient = red_target_gradient * (canvas_mask - 1) * (-1)
    #让白色变黑，黑色变白，白为1，可以得到各个通道背景梯度
    green_background_gradient = green_target_gradient * (canvas_mask - 1) * (-1)
    blue_background_gradient = blue_target_gradient * (canvas_mask - 1) * (-1)
    
    # add up foreground and background gradient
    gt_red_gradient = red_foreground_gradient + red_background_gradient#相加得到总的梯度
    gt_green_gradient = green_foreground_gradient + green_background_gradient
    gt_blue_gradient = blue_foreground_gradient + blue_background_gradient
    
    gt_red_gradient = numpy2tensor(gt_red_gradient, gpu_id)#变成tensor形式
    gt_green_gradient = numpy2tensor(gt_green_gradient, gpu_id)  
    gt_blue_gradient = numpy2tensor(gt_blue_gradient, gpu_id)
    
    gt_gradient = [gt_red_gradient, gt_green_gradient, gt_blue_gradient]
    return gt_gradient




class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False#有预训练参数 这里是false来节省时间

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

def gram_matrix(y):
    # gram matrix是一种计算向量间相似性的工具（target image和blend image之间的相似性）
    # 输入的张量y是一个四维张量，其中第一个维度是批次大小，第二个维度是通道数，第三个和第四个维度是特征图的高度和宽度。
    # 该函数将输入张量转换为（ b， ch， w * h），并计算其转置与自身的矩阵乘积。最后，该函数将结果除以通道数、高度和宽度的乘积

    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):#项目中未使用
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)#创建（3，1，1）的张量，其中每个元素都是一个标准差
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    #这些数是在ImageNet数据集上计算得出的
    #是所有图像的均值和标准差。这些值被用来规范化图像，以便在训练神经网络时获得更好的结果
    batch = batch.div_(255.0)
    #我们将输入批次除以255.0，
    # 然后从每个像素中减去均值，并将结果除以标准差。返回规范化后的批次
    return (batch - mean) / std



class MeanShift(nn.Conv2d):
    def __init__(self, gpu_id):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        rgb_range=1
        rgb_mean=(0.4488, 0.4371, 0.4040)
        rgb_std=(1.0, 1.0, 1.0)
        sign=-1
        #这里的sign=-1是为了将RGB值从[0, 1]的范围转换到[-1, 1]的范围。这个操作可以在训练过程中帮助模型更好地收敛。
        std = torch.Tensor(rgb_std).to(gpu_id)#std移到gpu上
        self.weight.data = torch.eye(3).view(3, 3, 1, 1).to(gpu_id) / std.view(3, 1, 1, 1)
        #eye（3）3x3的单位矩阵，即对角线上的元素为1
        #其中，`self.weight.data`是一个4维张量，
        # 其形状为`(out_channels, in_channels // groups, kernel_size[0], kernel_size[1])`。
        # `out_channels=in_channels=3`，`kernel_size=1`。
        # 因此，`self.weight.data`的形状为`(3, 3, 1, 1)`。这个张量是一个单位矩阵，除以了一个标准差向量的外积。
        # 标准差向量是由`rgb_std`定义的，它是一个长度为3的元组。这个张量的作用是将输入数据从RGB颜色空间转换到YUV颜色空间。
        #yuv空间类似于rgb空间，Y分量是 RGB三色的加权和， U、V 分量则可以视为 是亮度减去蓝、红
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean).to(gpu_id) / std
        for p in self.parameters():
            p.requires_grad = False#有预设参数，不需要更新


def get_matched_features_numpy(blended_features, target_features):
    matched_features = blended_features.new_full(size=blended_features.size(), fill_value=0, requires_grad=False)#与blendedfeature同尺寸的全0张量
    cpu_blended_features = blended_features.cpu().detach().numpy()#它是blended_features的CPU版本
    cpu_target_features = target_features.cpu().detach().numpy()
    for filter in range(0, blended_features.size(1)):
        matched_filter = torch.from_numpy(hist_match_numpy(cpu_blended_features[0, filter, :, :],
                                                           cpu_target_features[0, filter, :, :])).to(blended_features.device)
        #matched_filter是一个张量，
        # 它是通过hist_match_numpy函数将cpu_blended_features[0, filter, :, :]和cpu_target_features[0, filter, :, :]进行直方图匹配后得到的
        matched_features[0, filter, :, :] = matched_filter
    return matched_features


def get_matched_features_pytorch(blended_features, target_features):#以下同理
    matched_features = blended_features.new_full(size=blended_features.size(), fill_value=0, requires_grad=False).to(blended_features.device)
    for filter in range(0, blended_features.size(1)):
        matched_filter = hist_match_pytorch(blended_features[0, filter, :, :], target_features[0, filter, :, :])
        matched_features[0, filter, :, :] = matched_filter
    return matched_features


def hist_match_pytorch(source, template):

    oldshape = source.size()#源图像的尺寸
    source = source.view(-1)
    template = template.view(-1)#自适应尺寸

    max_val = max(source.max().item(), template.max().item())
    min_val = min(source.min().item(), template.min().item())
    #计算了输入张量source和template的最大值和最小值。这些值用于计算直方图的范围。

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins
    #计算直方图的范围，并将其分成400个区间。区间的大小为hist_step。

    if hist_step == 0:# <400时
        return source.reshape(oldshape)

    hist_bin_centers = torch.arange(start=min_val, end=max_val, step=hist_step).to(source.device)
    hist_bin_centers = hist_bin_centers + hist_step / 2.0
    # 计算直方图的中心。hist_bin_centers是一个张量，
    # 其中包含直方图中每个区间的中心值。这些值用于将一个直方图的值映射到另一个直方图

    source_hist = torch.histc(input=source, min=min_val, max=max_val, bins=num_bins)
    template_hist = torch.histc(input=template, min=min_val, max=max_val, bins=num_bins)
#   输入张量source和template的直方图。
    source_quantiles = torch.cumsum(input=source_hist, dim=0)
    source_quantiles = source_quantiles / source_quantiles[-1]
    #输入张量source的累积分布函数。它给出了一个随机变量小于或等于给定值的概率。用于将一个直方图的值映射到另一个直方图

    template_quantiles = torch.cumsum(input=template_hist, dim=0)
    template_quantiles = template_quantiles / template_quantiles[-1]
#    张量template的累积分布函数
    nearest_indices = torch.argmin(torch.abs(template_quantiles.repeat(len(source_quantiles), 1) - source_quantiles.view(-1, 1).repeat(1, len(template_quantiles))), dim=1)
#   输入张量source中每个元素在输入张量template中的最近邻索引
#   最近邻索引是一种算法，用于查找与给定查询项最相似的元素
    source_bin_index = torch.clamp(input=torch.round(source / hist_step), min=0, max=num_bins - 1).long()
#   将输入张量source中的每个元素四舍五入到最接近的整数，然后将其除以直方图步长。
#   这样可以将输入张量source中的每个元素映射到直方图中的一个bin
    mapped_indices = torch.gather(input=nearest_indices, dim=0, index=source_bin_index)
    #   使用输入张量source中的每个元素的索引来从输入张量nearest_indices中收集最近邻索引。
    matched_source = torch.gather(input=hist_bin_centers, dim=0, index=mapped_indices)
    #   使用mapped_indices中的索引从输入张量hist_bin_centers中收集元素
    return matched_source.reshape(oldshape)#变成oldshape的形状并返回


async def hist_match_pytorch_async(source, template, index, storage):

    oldshape = source.size()
    source = source.view(-1)
    template = template.view(-1)

    max_val = max(source.max().item(), template.max().item())
    min_val = min(source.min().item(), template.min().item())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        storage[0, index, :, :] = source.reshape(oldshape)
        return

    hist_bin_centers = torch.arange(start=min_val, end=max_val, step=hist_step).to(source.device)
    hist_bin_centers = hist_bin_centers + hist_step / 2.0

    source_hist = torch.histc(input=source, min=min_val, max=max_val, bins=num_bins)
    template_hist = torch.histc(input=template, min=min_val, max=max_val, bins=num_bins)

    source_quantiles = torch.cumsum(input=source_hist, dim=0)
    source_quantiles = source_quantiles / source_quantiles[-1]

    template_quantiles = torch.cumsum(input=template_hist, dim=0)
    template_quantiles = template_quantiles / template_quantiles[-1]

    nearest_indices = torch.argmin(torch.abs(template_quantiles.repeat(len(source_quantiles), 1) - source_quantiles.view(-1, 1).repeat(1, len(template_quantiles))), dim=1)

    source_bin_index = torch.clamp(input=torch.round(source / hist_step), min=0, max=num_bins - 1).long()

    mapped_indices = torch.gather(input=nearest_indices, dim=0, index=source_bin_index)
    matched_source = torch.gather(input=hist_bin_centers, dim=0, index=mapped_indices)

    storage[0, index, :, :] = matched_source.reshape(oldshape)
    #匹配的源张量重新形状为原始形状，并将其存储在给定的存储器中。这个存储器是一个张量，它的形状是 (1, num_images, num_channels, height, width)。
    #这个函数的目的是将所有图像的匹配源张量存储在一个大张量中，以便在训练过程中进行后处理。


async def loop_features_pytorch(source, target, storage):
    size = source.shape
    tasks = []

    for i in range(0, size[1]):
        task = asyncio.ensure_future(hist_match_pytorch_async(source[0, i], target[0, i], i, storage))
        tasks.append(task)

    await asyncio.gather(*tasks)#asyncio.gather() 函数来等待所有任务完成,收集匹配源张量


def get_matched_features_pytorch_async(source, target, matched):
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(loop_features_pytorch(source, target, matched))
    loop.run_until_complete(future)
    loop.close()
    #源张量、目标张量和存储器作为输入，并返回匹配的源张量。
    #该函数使用异步编程并发地调用 loop_features_pytorch 函数，以匹配每个源张量。
    #最后返回匹配的源张量。


def hist_match_numpy(source, template):

    oldshape = source.shape

    source = source.ravel()
    template = template.ravel()

    max_val = max(source.max(), template.max())
    min_val = min(source.min(), template.min())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        return source.reshape(oldshape)

    source_hist, source_bin_edges = np.histogram(a=source, bins=num_bins, range=(min_val, max_val))
    template_hist, template_bin_edges = np.histogram(a=template, bins=num_bins, range=(min_val, max_val))

    hist_bin_centers = source_bin_edges[:-1] + hist_step / 2.0

    source_quantiles = np.cumsum(source_hist).astype(np.float32)
    source_quantiles /= source_quantiles[-1]
    template_quantiles = np.cumsum(template_hist).astype(np.float32)
    template_quantiles /= template_quantiles[-1]

    index_function = np.vectorize(pyfunc=lambda x: np.argmin(np.abs(template_quantiles - x)))
    #用于查找最接近给定值的模板分位数
    #分位数是指在统计学中，将一组数据按大小顺序排列后分成等份的数值点，
    #这些数值点就是分位数。例如，中位数就是一组数据的50%分位数。

    nearest_indices = index_function(source_quantiles)

    source_data_bin_index = np.clip(a=np.round(source / hist_step), a_min=0, a_max=num_bins-1).astype(np.int32)

    mapped_indices = np.take(nearest_indices, source_data_bin_index)
    matched_source = np.take(hist_bin_centers, mapped_indices)

    return matched_source.reshape(oldshape)#返回匹配的源张量


def main():
    size = (64, 512, 512)
    source = np.random.randint(low=0, high=500000, size=size).astype(np.float32)
    target = np.random.randint(low=0, high=500000, size=size).astype(np.float32)
    source_tensor = torch.Tensor(source).to(0)
    target_tensor = torch.Tensor(target).to(0)
    matched_numpy = np.zeros(shape=size)
    matched_pytorch = torch.zeros(size=size, device=0)

    numpy_time = time.process_time()

    for i in range(0, size[0]):
        matched_numpy[i, :, :] = hist_match_numpy(source[i], target[i])
    
    numpy_time = time.process_time() - numpy_time

    pytorch_time = time.process_time()

    for i in range(0, size[0]):
        matched_pytorch[i, :, :] = hist_match_pytorch(source_tensor[i], target_tensor[i])
    
    pytorch_time = time.process_time() - pytorch_time


if __name__ == "__main__":
    main()
