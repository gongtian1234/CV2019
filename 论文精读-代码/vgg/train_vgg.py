import tensorflow as tf
import os
from VGG16 import VGG16
from load_data import *
from datetime import datetime
from tensorflow.data import Iterator
import numpy as np
import matplotlib.pyplot as plt

# 超参数设置
learning_rate = 0.001
decay_rate = 0.1
batch_size = 32
num_epochs = 100
rate = 0.5  # dropout的比率
filewriter_path = './tensorboard'
checkpoint_path = './checkpoints'
vgg16_npy_path = './vgg16.npy'
# 训练图像的路径
data_dir = './flower_photos'
num_classes = 5
# 将数据划分为训练集和测试集，给不同的花卉品种分配标签
training_filenames, validation_filenames, class_names_to_ids = split_datas(data_dir)
# 获取数据和相应的标签列表
train_data, train_label = get_datas_and_labels('train', training_filenames, class_names_to_ids)  # 图片列表以及该图片所对应的标签
val_data, val_label = get_datas_and_labels('validation', validation_filenames, class_names_to_ids)

# 加载batch_size数据，设备为cpu:0
with tf.device('/cpu:0'):
    train = Load_data(image_file=train_data, image_label=train_label, batch_size=batch_size, num_classes=num_classes,
                      mode='training', shuffle=True, buffer_size=100)
    val = Load_data(val_data, val_label, batch_size, num_classes, 'validation', shuffle=False, buffer_size=100)
    # create an reintializable iterator given the dataset structure
    iterator = Iterator.from_structure(train.data.output_types, train.data.output_shapes)
    next_batch = iterator.get_next()
# 分别给训练数据和验证数据初始化迭代器
training_init_op = iterator.make_initializer(train.data)
validation_int_op = iterator.make_initializer(val.data)

# 定义占位符
x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
fc_rate = tf.placeholder(tf.float32)   # 【问题】不明白这个是干嘛的，dropout不是在网络中已经传入了吗
# 载入网络
vgg16 = VGG16(vgg16_npy_path, False)   # 卷积层和池化层的参数不可训练
# vgg16.build(x, num_classes, dropout=rate)
vgg16.build(x, num_classes, dropout=fc_rate)   # 感觉上面那行代码应该有问题

# 将特征图写入tensorboard, 这里是将第一个feature map写入了tensorboard
with tf.variable_scope('feature_map'):
    conv1_1_feature = vgg16.conv1_1
    split_number = conv1_1_feature.get_shape().as_list()[-1]
    conv1_1 = tf.split(conv1_1_feature, num_or_size_splits=split_number, axis=3)
    tf.summary.image('conv1', conv1_1[0], max_outputs=4)

y_predict = vgg16.fc8
# 计算损失
vgg16.loss(y_predict, y)
# 调用优化器最小化损失
vgg16.training(learning_rate, decay_rate)
# 验证网络准确率
vgg16.evaluation(y_predict, y)
# 记录summary数据
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # 产生一个writer来写summary日志，并记录网络结构
    train_writer = tf.summary.FileWriter(filewriter_path, sess.graph)
    # 产生一个saver来持久化模型，默认保存最新的五个模型
    saver = tf.train.Saver()
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    # 自动恢复model_checkpoint_path保存的最新的模型
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored ...')
    else:
        print('No Model:start training')

    for epoch in range(num_epochs):
        steps = int(np.floor(len(training_filenames)/batch_size))   # np.floor()是向下取整
        print('One epoch = {} steps'.format(steps))
        sess.run(training_init_op)    # 初始化训练集的迭代器
        for step in range(steps):
            total_steps = (steps * epoch + step)
            img_batch, label_batch = sess.run(next_batch)
            _, train_loss = sess.run([vgg16.train_op, vgg16.cost], feed_dict={x:img_batch, y:label_batch, fc_rate:rate})
            # 迭代50次打印loss
            if total_steps%50==0:
                print('total steps {}:train loss:{:.5f}'.format(total_steps, train_loss))
                summary_str = sess.run(summary_op, feed_dict={x:img_batch, y:label_batch, fc_rate:rate})
                train_writer.add_summary(summary_str, step)
        sess.run(validation_int_op)   # 初始化测试集的迭代器
        # 一个epoch验证一次
        val_acc = 0.
        val_count = 0.
        for _ in range(int(np.floor(len(validation_filenames)/batch_size))):
            img_batch, label_batch = sess.run(next_batch)
            accuracy = sess.run(vgg16.accuracy, feed_dict={x: img_batch, y: label_batch, fc_rate:1})
            val_acc += accuracy
            val_count += 1
        val_acc /= val_count
        print('{} Validation Accuracy = {:.4f}'.format(datetime.now(), val_acc))
        print('{} Saving checkpoing of model ...'.format(datetime.now()))   # 每个epoch后都会保存一次模型

        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        save_path = os.path.join(checkpoint_path, 'model.ckpt')
        saver.save(sess, save_path, global_step=total_steps)
        print('{} Model checkpoint saved at {}'.format(datetime.now(), checkpoint_path))









