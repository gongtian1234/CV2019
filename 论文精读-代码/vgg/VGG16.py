import tensorflow as tf
import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]

class VGG16:
    def __init__(self, vgg16_npy_path=None, trainable=True):
        if vgg16_npy_path is not None:
            self.data_dict = np.load(vgg16_npy_path, allow_pickle=True, encoding='latin1').item()  # item()把字典的键值对组成元组
            del self.data_dict['fc8']  # 把全连接层删除
            del self.data_dict['fc7']
            del self.data_dict['fc6']
            print('====load weightts succesfully====')
        else:
            self.data_dict = None
        self.var_dict = {}
        self.trainable = trainable    # 在参数是否可训练处有用到

    def build(self, x, num_classes, dropout):
        # 第一个卷积块
        self.conv1_1 = self.conv_layer(x, in_channels=3, out_channels=64, name='conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, 'conv1_2')
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        # 第二个卷积块
        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, 'conv2_2')
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        # 第三个卷积块
        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        # 第四个卷积块
        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        # 第五个卷积块
        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        # 全连接层FC1
        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, 'fc6')  # 25088 = 7 * 7 * 512
        self.relu6 = tf.nn.relu(self.fc6)
        self.relu6 = tf.nn.dropout(self.relu6, keep_prob=dropout)
        # FC2
        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, 'fc7')
        self.relu7 = tf.nn.relu(self.fc7)
        self.relu7 = tf.nn.dropout(self.relu7, dropout)
        # FC3
        self.fc8 = self.fc_layer(self.relu7, 4096, num_classes, 'fc8')
        self.logits = self.fc8
        self.prop = tf.nn.softmax(self.logits, name='prob')
        self.data_dict = None

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_bias = self.get_conv_var(3, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filter=filt, strides=[1,1,1,1], padding='SAME', name=name)
            bias = tf.nn.bias_add(conv, bias=conv_bias)
            relu1 = tf.nn.relu(bias)
            return relu1

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal(shape=[filter_size, filter_size, in_channels, out_channels], stddev=0.001)
        filters = self.get_var(initial_value, name, 0, name+'_filters')

        initial_value = tf.truncated_normal([out_channels], stddev=0.0001)
        biases = self.get_var(initial_value, name, 1, name+'_biases')

        return filters, biases

    def get_var(self, initial_value, name, idx, var_name):  # idx: 权重为0，偏置为1
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value
        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)
        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()
        return var

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(value=bottom, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, shape=[-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal(shape=[in_size, out_size], stddev=0.0001)
        weights = tf.Variable(initial_value, name+'_weights')
        initial_value = tf.truncated_normal([out_size], stddev=0.0001)
        biases = tf.Variable(initial_value, name+'_biases')
        return weights, biases

    def save_npy(self, sess, npy_path='./vgg19-save.npy'):
        assert isinstance(sess, tf.Session)  # 判断两个类型是否相同
        data_dict = {}
        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out
        np.save(npy_path, data_dict)
        print(('file saved', npy_path))
        return npy_path

    def get_val_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v, get_shape().as_list)
        return count

    def loss(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)  # 它的标签是one-hot编码[[0,1,0],[1,0,0]...],
            cost = tf.reduce_mean(cross_entropy)
            # 标量记录
            tf.summary.scalar('loss', cost)
        self.cost = cost

    # 创建优化器
    def training(self, learning_rate, decay_rate):
        with tf.name_scope('optimizer'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,  # 指数衰减
                                                       decay_rate=decay_rate, decay_steps=2000, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(self.cost, global_step=global_step)
        self.train_op = train_op

    # 计算准确率
    def evaluation(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        self.accuracy = accuracy
