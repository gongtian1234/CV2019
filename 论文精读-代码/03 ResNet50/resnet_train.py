import keras
# from keras.applications.resnet50 import ResNet50
## 由于调用的是预训练的resnet网络，所以一般训练时会先下载预训练文件，
## 在这里，我是直接手动下载了预训练文件resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5，
## 然后又找到resnet50.py将其与权重文件一起放在这里；如果希望电脑自动下载预训练文件，可以直接把
## 下面的from resnet50 import ResNet50注释掉，解掉上面的那行注释
from resnet50 import ResNet50
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

# 实例化网络结构
resnet_50 = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
top_model = Sequential()
top_model.add(Flatten(input_shape=resnet_50.output_shape[1:], name='flatten'))
top_model.add(Dense(units=256, activation='relu',name='fc1'))
top_model.add(Dropout(rate=0.5, name='dropout1'))
top_model.add(Dense(units=54, activation='softmax', name='output_0'))

# 将两者拼接起来
model = Sequential()
model.add(resnet_50)
model.add(top_model)

# 模型的结构图：因为是拼接的，所以只会显示两层
# plot_model(model, to_file='resnet_50.png', show_shapes=True)

# # 图像增强操作只能放在model.compile()前面，如果放在后面，准确率不会有提升
train_datagen = image.ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                        shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                        fill_mode='nearest')  # rescale=1/255
val_datagen = image.ImageDataGenerator()

batch_size = 32
train_generator = train_datagen.flow_from_directory('image/train', target_size=(224,224),batch_size=batch_size)
val_generator = val_datagen.flow_from_directory('image/val', target_size=(224,224), batch_size=batch_size)

print(train_generator.class_indices)

model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=31,
                   validation_data=val_generator, validation_steps=len(val_generator))
# model.save('models/resnet50.h5')

# from save_model import save_pb_model
import tensorflow as tf
from keras import backend

def save_pb_model(model):
    ''' 只执行保存为pb的过程 '''
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input_img': model.input}, outputs={'output_score': model.output})
    builder = tf.saved_model.builder.SavedModelBuilder('models4')
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=backend.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images': signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()
    print('save pb to local path success')
    
save_pb_model(model)