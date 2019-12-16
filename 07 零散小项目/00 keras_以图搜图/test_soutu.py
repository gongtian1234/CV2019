# 进行测试
from main_soutu import *
import matplotlib.pyplot as plt

dirpath = '13大观楼/'
modelpath = 'models/vgg_featureCNN.h5'	
query = 'test.jpg'
queryimg = plt.imread(query)
plt.imshow(queryimg)
plt.title('Query Image.')
plt.pause(2)
plt.close()

h5f = h5py.File(modelpath, 'r')
feats = h5f['dataset_0'][:]
img_names = h5f['dataset_1'][:]
# print(feats[0].shape)   # shape为(512,)
# print(img_names)
h5f.close()

print('-'*30)
print('Searching...')
print('-'*30)

model = VGGNET16()
query_vec = model.extract_vgg_last_feat(query)    # shape为512
# 根据余弦相似度
scores = np.dot(query_vec, feats.T)    # shape为(64,)
rankId = np.argsort(scores)[::-1]      # 返回的是排序的从大到小排名的索引值
rank_score = scores[rankId]
# print(rank_score)

maxres = 5  # 相似图片的张数
imlist = [] 
for i, index in enumerate(rankId[0:maxres]):
	imlist.append(img_names[index])
	print('image name is {}, score is {}'.format(img_names[index], rank_score[i]))
print('top {} images in order are {}'.format(maxres, imlist))

# for i, img_path in enumerate(imlist):
	# tmp_img = plt.imread(dirpath)
	# plt.imshow(tmp_img)
	# plt.show()
