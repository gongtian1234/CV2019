1、inception_v1.py是用slim实现的具体的网络结构
2、04inceptionv3_test.ipynb是基于keras中的inceptionv3对猫狗数据进行的测试
3、模型的训练数据存放格式如下：
|image
----|train
    ----|1（表示类别，一个小类对应一个）
    ----|2
    ----|3
----|val
    ----|1
    ----|2
    ----|3

