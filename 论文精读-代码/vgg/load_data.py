import tensorflow as tf
import os
import random
import sys
from tensorflow.python.framework import dtypes
from tensorflow.data import Dataset
from tensorflow.python.framework.ops import convert_to_tensor
_RANDOM_SEED = 2019
_RATIO = 0.2   # valitation的比例


class Load_data(object):  # 一共有四个方法
    def __init__(self, image_file, image_label, batch_size, num_classes, mode, shuffle, buffer_size=1000):
        self.image_file = image_file
        self.image_label = image_label
        self.num_classes = num_classes
        self.img_paths = convert_to_tensor(self.image_file, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.image_label, dtype=dtypes.int32)
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))
        if mode=='training':
            data = data.map(self._parse_function_train, num_parallel_calls=4)   # 对训练集进行预处理, num_parallel_calls要并行处理的元素数
            data = data.prefetch(buffer_size=batch_size*100)
        elif mode=='validation':
            data = data.map(self._parse_function_val, num_parallel_calls=4)     # 对测试集进行预处理
            data = data.prefetch(buffer_size=batch_size*100)
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)
        data = data.batch(batch_size)
        self.data = data

    def _mean_image_subtraction(self, image):
        means = [123.69, 116.78, 103.69]
        if image.get_shape().ndims!=3:
            raise ValueError('Input must be of size[height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means)!=num_channels:
            raise ValueError('len(mean) must match the number of channels')
        channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=2, values=channels)

    def _parse_function_train(self, filename, label):
        one_hot = tf.one_hot(label, self.num_classes)
        # laod and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        image_data = tf.image.resize_images(img_decoded, [224, 224])   # 会使原始图像失真
        # data augmentation
        '''
        image_data = tf.image.random_flip_left_right(image_data)   # 左右翻转
        image_data = tf.image.random_brightness(image_data, 0.5)   # 图像亮度
        '''
        # 减去均值
        image_data = self._mean_image_subtraction(image_data)
        return image_data, one_hot

    def _parse_function_val(self, filename, label):
        one_hot = tf.one_hot(label, self.num_classes)
        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [224, 224])
        '''
        data augmentation come here
        '''
        img_data = self._mean_image_subtraction(img_resized)
        return img_data, one_hot


def _get_filenames_and_classes(dataset_dir):
    # 生成包含图像绝对路径的列表，以及花卉品种排序后的列表
    # :return: 返回一个列表这个列表包含了所有图片的路径，返回花卉种类名称
    directories, class_name = [], []
    for filename in os.listdir(dataset_dir):
        # 获取每个文件夹的绝对路径
        # directories=['./flower_photos/roses', './flower_photos/daisy', ...]
        # class_names=['roses', 'daisy', 'tulips', 'dandelion', 'sunflowers']
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_name.append(filename)
    photo_filenames = []
    # 获取每张图片的绝对路径，photo_filenames=['./flower_photos/roses/488849503_63a290a8c2_m.jpg',...]
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)
    return photo_filenames, sorted(class_name)


def split_datas(data_dir):
    photo_filenames, class_names = _get_filenames_and_classes(data_dir)
    # 给花卉分配标签{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    # 打乱数据
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    _NUM_VALIDATION = int(_RATIO*len(photo_filenames))
    training_filenames = photo_filenames[_NUM_VALIDATION:]
    validation_filenames = photo_filenames[:_NUM_VALIDATION]
    return training_filenames, validation_filenames, class_names_to_ids


def get_datas_and_labels(split_name, filenames, class_names_to_ids):
    '''

    :param split_name: train or validation
    :param filenames: 训练图像路径list，验证图像路径list
    :param class_names_to_ids: 花卉种类与label组成的字典
    :return: 图像路径list与对应的label
    '''
    assert split_name in ['train', 'validation']
    image_file, label_file = [], []
    for i in range(len(filenames)):
        class_name = os.path.basename(os.path.dirname(filenames[i]))
        class_id = class_names_to_ids[class_name]   # 通过key获取label信息
        image_file.append(filenames[i])
        label_file.append(class_id)
    return image_file, label_file


if __name__ == '__main__':
    data_dir = './flower_photos'
    training_filenames, validation_filenames, class_names_to_ids = split_datas(data_dir)
    train_data, train_label = get_datas_and_labels('train', training_filenames, class_names_to_ids)
    val_data, val_label = get_datas_and_labels('validation', validation_filenames, class_names_to_ids)
    train = Load_data(train_data, train_label, 1, 5, shuffle=True, buffer_size=1000)
    print(val_data)



