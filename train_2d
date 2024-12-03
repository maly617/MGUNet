import cv2
import os
import torch
import pandas
import time
import argparse
import numpy as np
from PIL import ImageFile
from torch.optim import lr_scheduler
from scipy.stats import pearsonr
from utils import DataProcess
from models.Coordgate_U import UNet
from utils import Operation
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

parser = argparse.ArgumentParser(description='Model Training With Pytorch')
parser.add_argument('--trainloss_dir', type=str, default='results/train.csv',
                    help='save train loss path')
parser.add_argument('--valloss_dir', type=str, default='results/test.csv',
                    help='save val loss path')
args = parser.parse_args()
model_dir_1 = 'E:/WorkSpace/logs/model.pth'
model_dir_2 = 'E:/WorkSpace/logs/'
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 为解决图像文件被截断问题，添加
train_demo = DataProcess.DataSetFunc('E:/WorkSpace/data/2d_image/list_train_1.csv',train_flag=True)  # 实例化一个对象, 训练集
val_demo = DataProcess.DataSetFunc('E:/WorkSpace/data/2d_image/list_val_1.csv')
test_demo = DataProcess.DataSetFunc('E:/WorkSpace/data/2d_image/list_test_1.csv')
lr = 0.0001  
net = UNet(1, 1)
criterion = torch.nn.MSELoss()
smooth_l1_loss = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.1)
# optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
# scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # 腹部406 几何249 brain 475*30
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=lr * 0.01)
# 判断能否使用GPU
use_gpu = torch.cuda.is_available()
if use_gpu:
    net = net.cuda()

# 模型训练
def train_model(model, epoch):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_mae = 0.0
    running_ssim = 0.0
    data_loader = torch.utils.data.DataLoader(train_demo, batch_size=16, shuffle=True, num_workers=0)  # 加载器
    for i, pack in enumerate(data_loader, 1):
        low_quality_img, label_img = pack
        if use_gpu:
            low_quality_img, label_img = torch.autograd.Variable(low_quality_img.float().cuda()), \
                                         torch.autograd.Variable(label_img.float().cuda())
        predict = model(low_quality_img)  #
        predict_1 = predict.detach().cpu().numpy().squeeze(1).transpose(1, 2, 0)
        label_img_1 = label_img.detach().cpu().numpy().squeeze(1).transpose(1, 2, 0)
        psnr = 0.0
        ssim = 0.0
        mae = 0.0
        for im_idx in range(label_img_1.shape[2]):
            psnr += Operation.calculate_psnr(predict_1[..., im_idx], label_img_1[..., im_idx])
            ssim += Operation.calculate_ssim(label_img_1[..., im_idx], predict_1[..., im_idx])
            mae += Operation.calculate_mae(label_img_1[..., im_idx], predict_1[..., im_idx])
        # loss = 0.6 * criterion(predict, label_img) + 0.4 * ((1 - ssim / 16) / 2)
        smooth_loss = smooth_l1_loss(predict, label_img)
        predict = predict.contiguous()
        loss = criterion(predict, label_img)  + 0.2 * smooth_loss
        # loss = combined_loss_fn(predict, label_img)
        # loss = criterion(predict, label_img)
        optimizer.zero_grad()
        loss.backward()  # 反向传播
        optimizer.step()  # Does the update
        running_loss += loss.data
        running_psnr += psnr
        running_mae += mae
        running_ssim += ssim
    epoch_loss = running_loss / train_demo.__len__()
    epoch_psnr = running_psnr / train_demo.__len__()
    epoch_ssim = running_ssim / train_demo.__len__()
    epoch_mae = running_mae / train_demo.__len__()
    scheduler.step()
    # with open(args.trainloss_dir, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(
    #         [str(epoch), str(epoch_loss), str(epoch_psnr), str(epoch_ssim), str(epoch_mae)])  # , str(epoch_ssim)
    print(epoch, epoch_loss, epoch_psnr, epoch_ssim, epoch_mae, scheduler.get_last_lr())  # epoch_ssim
    return epoch_ssim

# 模型测试
def detection_model_csv(epoch, model):
    model.eval()
    running_psnr = 0.0
    running_mae = 0.0
    running_ssim = 0.0
    running_p_sum = 0.0
    data_loader = torch.utils.data.DataLoader(val_demo, batch_size=1, shuffle=False, num_workers=0)  # 加载器
    for i, pack in enumerate(data_loader, 1):
        low_quality_img, label_img = pack
        if use_gpu:
            low_quality_img, label_img = torch.autograd.Variable(low_quality_img.float().cuda()), \
                                         torch.autograd.Variable(label_img.float().cuda())

        predict = model.forward(low_quality_img)  # , pre2
        predict_1 = predict.detach().cpu().numpy().squeeze(1).transpose(1, 2, 0)
        label_img_1 = label_img.detach().cpu().numpy().squeeze(1).transpose(1, 2, 0)
        psnr = 0.0
        ssim = 0.0
        mae = 0.0
        p_sum = 0.0
        for im_idx in range(label_img_1.shape[2]):
            psnr += Operation.calculate_psnr(predict_1[..., im_idx], label_img_1[..., im_idx])
            ssim += Operation.calculate_ssim(label_img_1[..., im_idx], predict_1[..., im_idx])
            mae += Operation.calculate_mae(label_img_1[..., im_idx], predict_1[..., im_idx])
            img1_flattened = label_img_1[..., im_idx].flatten()
            img2_flattened = predict_1[..., im_idx].flatten()
            pearson_coefficient, _ = pearsonr(img1_flattened, img2_flattened)
            p_sum += pearson_coefficient
        running_psnr += psnr
        running_mae += mae
        running_ssim += ssim
        running_p_sum += p_sum
    epoch_psnr = running_psnr / val_demo.__len__()
    epoch_mae = running_mae / val_demo.__len__()
    epoch_ssim = running_ssim / val_demo.__len__()
    epoch_psum = running_p_sum / val_demo.__len__()
    # with open(args.valloss_dir, 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([str(epoch), str(epoch_psnr), str(epoch_ssim), str(epoch_mae), str(epoch_psum)])
    print(epoch, epoch_psnr, epoch_ssim, epoch_mae, epoch_psum)
    return epoch_ssim

def main():
    best_metric = 0.75
    torch.cuda.empty_cache()
    if os.path.exists(model_dir_1):
        checkpoint = torch.load(model_dir_1)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
       
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')
    time_open = time.time()
    for epoch in range(start_epoch, 150):
        temp_ssim_q = train_model(net, epoch)  # 训练模型
        temp_ssim = detection_model_csv(epoch, net)  # 验证模型
        if temp_ssim >= best_metric:
            best_metric = temp_ssim
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}  # 保存模型
            model_dir_new = model_dir_2 + 'model_' + str(epoch) + '.pth'
            torch.save(state, model_dir_new)
        else:
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}  # 保存模型
            model_dir_new = model_dir_2 + 'model_' + '.pth'
            torch.save(state, model_dir_new)
        # detection_model_csv(epoch, net)  # 验证模型
    time_end = time.time() - time_open
    print(time_end)  # 输出训练总耗时
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
