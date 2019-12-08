前期准备工作：
①!wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar下载并解压至data/pascal_voc文件夹下


1、yolonet_v1.py包括了网络的建立和loss函数；

2、config.py用于初始化参数

3、train.py为主程序，训练网络
如果要训练网络，直接运行train.py即可

4、pascal_voc.py产生训练数据和label，主要用于对数据进行预处理

5、timer.py封装的计时模块

数据文件夹：(具体看https://blog.csdn.net/xiaotian127/article/details/103405577)
./data
     |pascal_voc
         |VOCdevkit
             |VOC2007
                 |Annotations（存放的是每张图片对应的xml文件）
                 |JPEGImages（存放的是图片）
                 |ImageSets
     |weights（如果是跑官方代码的话，这里放的是YOLO_small.ckpt）
