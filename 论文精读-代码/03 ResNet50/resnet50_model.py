from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend, initializers, layers, models, regularizers
import imagenet_preprocessing

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILOW = 1e-5


def _gen_l2_regularizer(use_l2_regularizer=True):
    return regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2), use_l2_regualizer=True):
    '''
    :param filters:list of integers, the filters of 3 conv layer at main path
    :param stage: integer, current stage label, used for generating layer names
    :param block: 'a','b'..., current block label, used for generating layer names
    '''
    filters1, filters2, filters3 = filters
    if backend.image_data_format()=='channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1,1), use_bias=False, kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(use_l2_regualizer),
                      name=conv_name_base+'2a')(input_tensor)
    x = layers.BatchNormalization(bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILOW, name=bn_name_base+'2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, strides=strides, padding='same', use_bias=False,
                      kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(use_l2_regualizer),
                      name=conv_name_base+'2b')
    x = layers.BatchNormalization(bn_axis, BATCH_NORM_DECAY, BATCH_NORM_EPSILOW, name=bn_name_base+'2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1,1), use_bias=False, kernel_regularizer=_gen_l2_regularizer(use_l2_regualizer),
                      kernel_initializer='he_normal', name=conv_name_base+'2c')(x)
    x = layers.BatchNormalization(bn_axis, BATCH_NORM_DECAY, BATCH_NORM_EPSILOW, name=bn_name_base+'2c')(x)

    shortcut = layers.Conv2D(filters3, (1,1), strides=strides, use_bias=False, kernel_initializer='he_normal',
                             kernel_regularizer=_gen_l2_regularizer(use_l2_regualizer), name=conv_name_base+'1')(input_tensor)
    shortcut = layers.BatchNormalization(bn_axis, BATCH_NORM_DECAY, BATCH_NORM_EPSILOW, name=bn_norm_base+'1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block, use_l2_regularizer=True):
    filters1, filters2, filters3 = filters
    if backend.image_data_format()=='channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1,1), use_bias=False, kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                      name=conv_name_base+'2a')(input_tensor)
    x = layers.BatchNormalization(bn_axis, BATCH_NORM_DECAY, BATCH_NORM_EPSILOW, name=bn_name_base+'2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), name=conv_name_base+'2b')(x)
    x = layers.BatchNormalization(bn_axis, BATCH_NORM_DECAY, BATCH_NORM_EPSILOW, name=bn_name_base+'2b')
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1,1), use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), name=conv_name_base+'2c')(x)
    x = layers.BatchNormalization(bn_axis, BATCH_NORM_DECAY, BATCH_NORM_EPSILOW, name=bn_name_base+'2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def resnet50(num_classes, batch_size=None, use_l2_regularizer=True, rescale_inputs=False):
    input_shape = (224, 224, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    if rescale_inputs:
        x = layers.Lambda(
            lambda x: x*255.0-backend.constant(imagenet_preprocessing.CHANNEL_MEANS, shape=[1,1,3], dtype=x.dtype, name='rescale')(img_input)
        )
    else:
        x = img_input

    if backend.image_data_format()=='channels_first':
        x = layers.Lambda(lambda x:backend.permute_dimensions(x, (0,3,1,2)), name='tranpose')(x)
        bn_axis = 1
    else:  # channels_last
        bn_axis = 3

    x = layers.ZeroPadding2D(padding=(3,3), name='conv1_pad')(x)    # 在上下左右各填充3个尺寸 :230
    x = layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='valid', use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), name='conv1')(x)  # :224
    x = layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILOW, name='bn_conv1')
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(x)  # :224

    x = conv_block(input_tensor=x, kernel_size=3, filters=[64,64,256], stage=2, block='a',
                   strides=(1,1), use_l2_regualizer=use_l2_regularizer)  # :224

    x = identity_block(input_tensor=x, kernel_size=3, filters=[64,64,256], stage=2, block='b', use_l2_regularizer=use_l2_regularizer)  # :224

    x = identity_block(x, 3, [64,64,256], stage=2, block='c', use_l2_regularizer=use_l2_regularizer)

    x = conv_block(x, 3, [128,128,512], stage=3, block='a', use_l2_regualizer=use_l2_regularizer)

    x = identity_block(x, 3, [128,128,256], stage=3, block='b', use_l2_regularizer=use_l2_regularizer)

    x = identity_block(x, 3, [128,128,256],stage=3, block='c', use_l2_regularizer=use_l2_regularizer)

    x = identity_block(x, 3, [128,128,256],stage=3, block='d', use_l2_regularizer=use_l2_regularizer)

    x = conv_block(x, 3, filters=[256,256,1024], stage=4, block='a', use_l2_regualizer=use_l2_regularizer)

    x = identity_block(x, 3, [256,256,1024], stage=4, block='b', use_l2_regularizer=use_l2_regularizer)

    x = identity_block(x, 3, [256,256,1024], stage=4, block='c', use_l2_regularizer=use_l2_regularizer)

    x = identity_block(x, 3, [256,256,1024], stage=4, block='d', use_l2_regularizer=use_l2_regularizer)

    x = identity_block(x, 3, [256,256,1024], stage=4, block='e', use_l2_regularizer=use_l2_regularizer)

    x = identity_block(x, 3, [256,256,1024], stage=4, block='f', use_l2_regularizer=use_l2_regularizer)

    x = conv_block(x, 3, [512,512,2048], stage=5, block='a', use_l2_regualizer=use_l2_regularizer)

    x = identity_block(x, 3, [512,512,2048], stage=5, block='b', use_l2_regularizer=use_l2_regularizer)

    x = identity_block(x, 3, [512,512,2048], stage=5, block='c', use_l2_regularizer=use_l2_regularizer)

    rm_axes = [1,2] if backend.image_data_format()=='channels_last' else [2,3]   # 平均池化层
    x = layers.Lambda(lambda x:backend.mean(x, rm_axes), name='reduce_mean')(x)

    x = layers.Dense(units=num_classes, kernel_initializer=initializers.RandomNormal(stddev=0.01),
                     kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                     bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                     name='fc1000')(x)
    x = layers.Activation('softmax', dtype='float32')(x)

    return models.Model(img_input, x, name='resnet50')


