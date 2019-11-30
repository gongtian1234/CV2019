from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v1_base(inputs, final_endpoint='Mixed_5c', scope='InceptionV1'):
    '''
    定义inceptionv1的基本网络结构
    :param final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    :param scope: Optional variable_scope.
    '''
    end_points = {}
    with tf.variable_scope(name_or_scope=scope, default_name='InceptionV1', values=[inputs]):
        with slim.arg_scope(list_ops_or_scope=[slim.conv2d, slim.fully_connected], weights_initializer=trunc_normal(0.01)):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], stride=1, padding='SAME'):
                end_point = 'Conv2d_1a_7x7'
                net = slim.conv2d(inputs=inputs, num_outputs=64, kernerl_size=[7, 7], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points
                end_point = 'MaxPool_2a_3x3'
                net = slim.max_pool2d(inputs=net, kernerl_size=[3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points
                end_point = 'Conv2d_2b_1x1'
                net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[1, 1], scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points
                end_point = 'MaxPool_2c_3x3'
                net = slim.conv2d(inputs=net, num_outputs=192, kernel_size=[3, 3], scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points
                end_point = 'MaxPool_3a_3x3'
                net = slim.max_pool2d(inputs=net, kernel_size=[3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_3b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope(name_or_scope='Branch_0'):
                        branch_0 = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope(name_or_scope='Branch_1'):
                        branch_1 = slim.conv2d(inputs=net, num_outputs=96, kernel_size=[1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(inputs=branch_1, num_outputs=128, kernel_size=[3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope(name_or_scope='Branch_2'):
                        branch_2 = slim.conv2d(inputs=net, num_outputs=16, kernel_size=[1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(inputs=branch_2, num_outputs=32, kernel_size=[3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope(name_or_scope='Branch_3'):
                        branch_3 = slim.max_pool2d(inputs=net, kernel_size=[3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, num_outputs=32, kernel_size=[1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_3c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(inputs=branch_1, num_outputs=192, kernel_size=[3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(inputs=net, num_outputs=32, kernel_size=[1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(inputs=branch_2, num_outputs=96, kernel_size=[3, 3], scope='Conv2d_0b_1x1')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(inputs=net, kernel_size=[3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(inputs=branch_3, num_outputs=64, kernel_size=[1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'MaxPool_4a_3x3'
                net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_4b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_4c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_4d'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_4e'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_4f'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'MaxPool_5a_2x2'
                net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_5b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points

                end_point = 'Mixed_5c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if final_endpoint == end_point: return net, end_points
        raise ValueError('Unknown final endpoint {}'.format(final_endpoint))


def inceptioin_v1(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.8,
                  prediction_fn=slim.softmax, spatial_squeeze=True, reuse=None, scope='InceptionV1'):
    '''
    定义inceptionv1的结构
    :param inputs: [b,h,w,c]
    :param is_training: whether is training or not
    :param spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
                            of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    :param reuse: whether or not the network and its variables should be reused. To be able to reuse 'scope' must be given.
    :param scope: Optional variable_scope.
    :return:
    '''
    with tf.variable_scope(name_or_scope=scope, default_name='InceptionV1',values=[inputs, num_classes], reuse=reuse) as scope:
        with slim.arg_scope(list_ops_or_scope=[slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = inception_v1_base(inputs, scope=scope)
            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(inputs=net, kernel_size=[7, 7], stride=1, scope='MaxPool_0a_7x7')
                net = slim.dropout(inputs=net, keep_prob=dropout_keep_prob, scope='Dropout_0b')
                logits = slim.conv2d(inputs=net, num_outputs=num_classes, kernel_size=[1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_0c_1x1')
                if spttial_squeeze:
                    logits = tf.squeeze(input=logits, axis=[1, 2], name='SpatialSqueeze')

                end_points['Logits'] = logits
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
        return logits, end_points


def inception_va_arg_scope(weight_decay=0.00004, use_batch_norm=True, batch_norm_val_collection='moving_vars'):
    '''
    Note: Althougth the original paper didn't use batch_norm we found it useful.
    :param weight_decay: The weight decay to use for regularizing the model.
    :param use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    :param batch_norm_val_collection: The name of the collection for the batch norm variables.
    '''
    batch_norm_params = {'decay': 0.9997,
                         'epsilon': 0.001,
                         'updates_collections': tf.GraphKeys.UPDATE_OPS,
                         'variables_collections': {'beta': None, 'gamma': None, 'moving_mean': [batch_norm_val_collection],
                                                   'moving_variance': [batch_norm_val_collection],}
                         }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    # Set weight_decay for weights in Conv and FC layers
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], weight_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu, normalizer_fn=normalizer_fn, normalizer_params=normalizer_params) as sc:
            return sc

