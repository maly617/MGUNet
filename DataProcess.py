import random
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CustomPairedTransform:
    def __init__(self, original_size=512, output_size=128):
        self.original_size = original_size
        self.output_size = output_size

    def __call__(self, img, label, train):

        if random.random() > 0.5:
           img = transforms.functional.hflip(img)
           label = transforms.functional.hflip(label)
        if random.random() > 0.5:
            img = transforms.functional.vflip(img)
            label = transforms.functional.vflip(label)
        # 随机亮度调整
        if train:
            brightness_factor = random.uniform(0.95, 1.05)
            img = transforms.functional.adjust_brightness(img, brightness_factor)

        # 随机选择裁剪
        max_x = self.original_size - self.output_size
        max_y = self.original_size - self.output_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        x = 128
        y = 128
        # 执行相同的随机裁剪
        img = transforms.functional.crop(img, y, x, self.output_size, self.output_size)
        label = transforms.functional.crop(label, y, x, self.output_size, self.output_size)
        # 转换为张量
        img = transforms.ToTensor()(img)
        label = transforms.ToTensor()(label)
        return img, label


class DataSetFunc(Dataset):
    def __init__(self, csv_path, train_flag=False):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            transform: transform 操作
        """
        # 读取 csv 文件
        self.data_info = pd.read_csv(csv_path, header=None, skiprows=0)
        self.train = train_flag
        self.to_tensor = CustomPairedTransform()
        # 第1列是图像的路径
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # 第2列是图像的 label 的名称
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # 计算 length
        self.data_len = len(self.data_info.index)
    def __getitem__(self, index):
        # 从 pandas df 中得到文件名
        single_image_name = self.image_arr[index]
        singel_label_name = self.label_arr[index]
        # 读取图像文件
        img_as_img = Image.open(single_image_name).convert('L')
        label_as_img = Image.open(singel_label_name).convert('L')
        # 应用配对变换
        img_as_tensor, label_as_tensor = self.to_tensor(img_as_img, label_as_img, self.train)
        return img_as_tensor, label_as_tensor

    def __len__(self):
        return self.data_len
