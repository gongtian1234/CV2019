--------------------------
tensorflow版本
--------------------------

官方地址：https://github.com/shekkizh/FCN.tensorflow

参考网址：
1、https://blog.csdn.net/m0_37407756/article/details/83379026（详细讲解了怎么从官方代码到训练自己的数据集）
2、https://blog.csdn.net/MOU_IT/article/details/81073149（跟第一篇差不多，把官方代码复制了一遍）

----------------------------------------------------------------------------------
通过batch_json_to_dataset.py、batch_color_map.py、3to1_channel.py进行图像的处理：
batch_json_to_dataset.py: 来自参考文章1里作者提供的下载文件，这个文件主要是批量将json文件（用labelme标注图像得到的json）转换为png图像（转换完看起来是黑乎乎的一片）；
python batch_json_to_dataset.py json(存放json的文件夹) gt_png3(output位置)

batch_color_map.py: 将上面转换为黑乎乎的图片进行着色，使其能够看出标注的图片，着色完图像是三通道的；所以这些图片不能直接使用；

3to1_channel.py：对上面生成的彩色三通道图片进行转换，将其转换为1通道的

----------------------------------------------------------------------------------
gen_pickle.py：将上面处理好的文件，存放为Data_zoo/MIT_ScenceParsing/官方代码的格式，然后利用这个文件生成对应的pickle文件：
文件夹格式为：
./Data_zoo/MIT_ScenceParsing|
                             |ADEChallengeData2016|
                                                   |annotations|
                                                                |training
                                                                |validation
                                                   |images|
                                                           |training
                                                           |validation
                             |MITSceneParsing.pickle（里面主要是存放文件的路径）

----------------------------------------------------------------------------------

FCN.py: 删除【NUM_OF_CLASSES改为类别数+1(背景类)；】，好像NUM_OF_CLASSES不用改，因为FCN中是对像素进行分类，反而如果改小了程序会一直报错。
