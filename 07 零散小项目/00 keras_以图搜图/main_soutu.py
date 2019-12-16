
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import h5py


class VGGNET16:
	'''定义一个类：用于提取特征'''
	def __init__(self):
		self.input_shape = (224,224,3)
		self.weights = 'imagenet'
		self.pooling = 'max'
		self.vgg_model = VGG16(input_shape=self.input_shape, weights=self.weights, 
						  include_top=False, pooling=self.pooling)
		self.vgg_model.predict(np.zeros((1,224,224,3)))

	def extract_vgg_last_feat(self, img_path):
		img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		feats = self.vgg_model.predict(x)    # shape为(512,)，因为取得是取得是vgg16的倒数第二层嘛
		norm_feats = feats[0] / np.linalg.norm(feats[0])    # 求范数，默认为ord=2，即求的是二阶范数
		return norm_feats

def get_imglist(dirpath):
	'''获取一个文件夹下的所有图片'''
	return [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith('.jpg')]

def save_extract_result(outputpath, content_list):
	h5f = h5py.File(outputpath, 'w')
	for i,contenti in enumerate(content_list):
		print('dataset_'+str(i), ' is saving...')
		h5f.create_dataset('dataset_'+str(i), data=contenti)
	h5f.close()

if __name__=='__main__':
	# dirpath = '1仿宋木叶盏/'
	dirpath = '13大观楼/'
	if not os.path.exists('models'):
		print('Create dir: models')
		os.makedirs('models')
	modelpath = 'models/vgg_featureCNN.h5'
	if not os.path.exists(modelpath):   # 防止重复生成已有的h5文件

		img_list = get_imglist(dirpath)

		print('-'*30)
		print('feature extract start...')
		print('-'*30)

		features = []
		names = []

		model = VGGNET16()
		for i ,img_path in enumerate(img_list):
			norm_feats = model.extract_vgg_last_feat(img_path)
			img_name = os.path.split(img_path)[1]  # img_path.split('.')[0]
			features.append(norm_feats)
			names.append(img_name)
			print('Extract feature from No.{}/{}'.format(i, len(img_list)))

		features = np.array(features)
		output = modelpath

		print('-'*30)
		print('Saving feature extract result...')
		print('-'*30)

		print(names)
		print(np.string_(names))
		save_extract_result(output, [features, np.string_(names)])

	

