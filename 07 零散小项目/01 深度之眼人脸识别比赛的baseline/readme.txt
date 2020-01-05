1、tensorflow基础.ipynb：
该文档详细的介绍了利用tensorlfow搭建模型的基础内容，如常量、变量(tf.Variable()/tf.get_variable('该函数不能定义重复的变量名')、初始化)、占位符、图(清空图、定义多个图等)等等。也实操了minist数据集；

2、keras基础.ipynb：
该文档详细的介绍了keras的接口，包括自定义评估函数、keras.models下的两种模型(Sequential API、Function API：可以叠加两层的输出作为下一层的输入)等，但是没有介绍大量数据时数据如何传入模型。也介绍了一些/site-packages/keras_applications/下已预训练好的模型文件

3、数据准备和增强.ipynb：
介绍了数据存储的几种方法(转为tfrecords、仍保持图片形式)、tf与keras混合使用训练mnist数据集的例子

4、构建baseline模型.ipynb: 
人脸识别的代码，先用face_recognition将每张图中的人脸抠出来并存储起来，再将这些人脸图片转换为tfrecords格式并进行相应的图像增强处理等操作，最后再用生成器来迭代训练模型。不足：没有对影响模型效果的难例进行分析，没涉及到这方面的trick