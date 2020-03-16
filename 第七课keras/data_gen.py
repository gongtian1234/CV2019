# -*- coding: utf-8 -*-
import os
import math
import codecs
import random
import numpy as np
from glob import glob
from PIL import Image, ImageFilter, ImageEnhance
import random

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
# import keras
# from keras_applications.imagenet_utils import preprocess_input
from keras.utils import np_utils, Sequence
from sklearn.model_selection import StratifiedShuffleSplit


class BaseSequence(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """
    def __init__(self, img_paths, labels, batch_size, img_size):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size
        if self.img_size[0]%2!=0:
            print('请使输入的img_size为一个偶数！！！')
            raise 

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    def preprocess_img(self, img_path):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        s = 256
        if self.img_size[0]>s:
            s = self.img_size[0] + 32 #
        # print('等比缩放的最小边为：',s)
        crop_size = self.img_size[0]    # 剪裁的尺寸是224*224
        img = image.load_img(img_path)  # 加载图片
        # img = Image.open(img_path)
        img_w, img_h = img.size         # 获取原始图像的宽和高
        scale = s/min(img_w, img_h)     # 缩放比例
        new_w, new_h = np.int(img_w*scale), np.int(img_h*scale)
        resize_img = img.resize((new_w, new_h))

        if random.random()<0.5:   # 50%的概率进行左右翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random()<0.5:   # 50%的概率进行通道交换
            r, g, b = img.split()
            rnum = random.random()
            if rnum<0.33:  # 33%的概率交换为r,b,g;  b,r,g;  g,r,b
                img = Image.merge('RGB', [r,b,g])
            elif rnum>0.66:
                img = Image.merge('RGB', [b,r,g])
            else:
                img = Image.merge('RGB', [g,r,b])
        if random.random()<0.5:  # 50%的概率应用高斯滤波器
            img = img.filter(ImageFilter.GaussianBlur)
        # img = img.point(lambda x: x*0.3)   # 这一步等价于图像的亮度操作
        if random.random()<0.65:   # 亮度变化 
            im_brightness = ImageEnhance.Brightness(img)
            img = im_brightness.enhance(random.uniform(0.7, 1.3))
        if random.random()<0.5:   # 对比度contrast变换,对比度越大越像是看得清晰，越接近于0，越模糊
            im_brightness = ImageEnhance.Contrast(img)
            img = im_brightness.enhance(random.uniform(0.7, 4))
        if random.random()<0.5:
            im_brightness = ImageEnhance.Color(img)
            img = im_brightness.enhance(random.uniform(0.7, 3))
        if random.random()<0.65:   # 进行图象旋转
            img = img.rotate(angle=random.uniform(-25, 25))
        
        max_offset_w = np.random.randint(low=0, high=new_w-crop_size+1, dtype='int32')
        max_offset_h = np.random.randint(low=0, high=new_h-crop_size+1, dtype='int32')
        crop_img = resize_img.crop((max_offset_w, max_offset_h,max_offset_w+crop_size, max_offset_h+crop_size))

        x = image.img_to_array(img=crop_img)
        # if random.random()<0.5:    # 50%的概率进行左右翻转
        #     x = np.flip(x,1)
        # x = np.asarray(crop_img, dtype=np.float32)
        x = preprocess_input(x)
        # img = image.load_img(img_path, target_size=(self.img_size[0], self.img_size[1]))
        # img = image.random_brightness(img, [0,2])   # 修改9：随机调亮度
        # if random.random()>0.7:
        #     img = image.image.flip_axis(img, axis=1)     # 修改9：0.5的概率水平翻转

        # img = image.load_img(img_path, target_size=(self.img_size[0], self.img_size[1]))
        # x = image.img_to_array(img)
        # x = preprocess_input(x)
        return x

    def __getitem__(self, idx):
        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]
        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)
        return batch_x, batch_y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        np.random.shuffle(self.x_y)

class BaseSequence_val(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """
    def __init__(self, img_paths, labels, batch_size, img_size):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size
        if self.img_size[0]%2!=0:
            print('请使输入的img_size为一个偶数！！！')
            raise 

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    def preprocess_img(self, img_path):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        s = 256
        if self.img_size[0]>s:
            s = self.img_size[0] + 32 #
        # print('等比缩放的最小边为：',s)
        crop_size = self.img_size[0]    # 剪裁的尺寸是224*224
        img = image.load_img(img_path)  # 加载图片
        img_w, img_h = img.size         # 获取原始图像的宽和高
        scale = s/min(img_w, img_h)     # 缩放比例
        new_w, new_h = np.int(img_w*scale), np.int(img_h*scale)
        zhongxin = (new_w//2, new_h//2)
        resize_img = img.resize((new_w, new_h))
        crop_img = resize_img.crop((zhongxin[0]-crop_size/2,zhongxin[1]-crop_size/2, zhongxin[0]+crop_size/2,zhongxin[1]+crop_size/2))
        # print('val剪裁完的图像尺寸为：',crop_img.size)
        x = image.img_to_array(img=crop_img)
        x = preprocess_input(x)
        return x

    def __getitem__(self, idx):
        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]
        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)
        return batch_x, batch_y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        np.random.shuffle(self.x_y)

def data_flow(train_data_dir, batch_size, num_classes, input_size, val_num=5000):
    label_files = glob(os.path.join(train_data_dir, '*.txt'))
    random.shuffle(label_files)
    '''
    img_paths = []
    labels = []
    for index, file_path in enumerate(label_files):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        img_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(train_data_dir, img_name))
        labels.append(label)
    '''
    info =  {
        "0": "cat",
        "1": "dog",
    }
    img_paths = []   # 存放图片路径
    labels = []# 存放其对应的类别标签
    # for tmp in ['train/','val/']:
    for tmp in ['train/','test/']:
        root_path = os.path.join(train_data_dir, tmp)
        for file_dir in os.listdir(root_path): # file_dir：文件夹的名字
            class_num = -1    # 先将文件夹对应的编号初始化为-1，方便最后检查，如果检查时有-1，则表明有文件夹没转换成功
            for classi, infoi in enumerate(info.values()):   # 主要是为了获取文件夹所对应的类别编号classi
                if file_dir in infoi:
                    print('{} 对应的编号为 {}'.format(file_dir, classi))
                    class_num = classi
            for home, dirs, files in os.walk(os.path.join(root_path,file_dir)):  # 开始遍历每一个文件夹
                for filename in files:
                    img_paths.append(os.path.join(home, filename))
                    labels.append(class_num)

    print('='*40)
    # print('img_paths:', img_paths)
    print()

    print('='*40)
    # print('labels:', labels)
    print()
    print('='*40)
    
    img_paths = np.array(img_paths)
    labels = np_utils.to_categorical(labels, num_classes)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_num, random_state=0)  # 您可以根据自己的需要调整 test_size 的大小
    sps = sss.split(img_paths, labels)
    for sp in sps:
        train_index, validation_index = sp
    print('total samples: %d, training samples: %d, validation samples: %d'
          % (len(img_paths), len(train_index), len(validation_index)))
    train_img_paths = img_paths[train_index]
    validation_img_paths = img_paths[validation_index]
    train_labels = labels[train_index]
    validation_labels = labels[validation_index]

    # 训练集是随机剪裁目标大小
    train_sequence = BaseSequence(train_img_paths, train_labels, batch_size, [input_size, input_size])
    # 验证集是中心剪裁
    validation_sequence = BaseSequence_val(validation_img_paths, validation_labels, batch_size, [input_size, input_size])

    return train_sequence, validation_sequence
