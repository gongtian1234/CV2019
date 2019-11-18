import tensorflow as tf
import numpy as np
from VGG16 import VGG16

VGG_MEAN = [103.939, 116.779, 123.68]
filename = './test_images/sunflowrs.jpeg'
checkpoint_path = './checkpoints'
vgg16_npy_path = './vgg16.npy'

img_string = tf.read_file(filename)
img_decoded = tf.image.decode_jpeg(img_string, channels=3)
image_data = tf.image.resize_images(img_decoded, [224, 224])

# 减去均值
def _mean_image_subtraction(image):
    means = [123.69, 116.78, 103.69]
    if image.get_shape().ndims!=3:
        raise ValueError('Input must be of size[height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means)!=num_channels:
        raise ValueError('len(means) nust match the number of channels')
    channels = tf.split(image, num_or_size_splits=num_channels, axis=2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


# 扩维
image = tf.expand_dims(_mean_image_subtraction(image_data), axis=0)

with tf.Session() as sess:
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    fc_rate = tf.placeholder(tf.float32)
    image = sess.run(image)
    feed_dict = {images: image, fc_rate: 1.0}
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    vgg = VGG16(vgg16_npy_path, trainable=False)
    vgg.build(images, 5, fc_rate)
    saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored ... fc6, fc7, fc8')
    prob = sess.run(vgg.prop, feed_dict=feed_dict)
    print(prob)
