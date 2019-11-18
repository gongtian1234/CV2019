代码结构：
train_vgg.py: 主程序，包含了整个工程的完整流程

load_data.py: 数据产生程序，包含了数据的读取与去均值过程
数据存储方式：data下每一个品种对应一个文件夹；
读取方式：将每类的文件夹名转换为对于那个的class，所有的图片的绝对路径都存在一个列表中；
数据集划分：随机打乱，按照一定的比例划分训练集合测试集；
又通过get_datas_and_labels()加载出训练集和测试集对应图片的绝对路径及对应的标签；
_mean_image_subtraction：图像减去均值的操作；
_parse_function_train：训练集的图像进行预处理（resize、减去均值、水平翻转、调整图像亮度等）；


VGG16.py：定义了网络结构、损失函数、准确率计算等；

test_vgg16.py: 测试模型效果

test_images文件夹：包含了若干张测试图片（来源于网络下载）

具体压缩包请参考百度网盘链接：
https://pan.baidu.com/s/10fM233DdwB5r73DSgJ3irQ 提取码：7g7e
