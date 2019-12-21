import os
import cv2

root_dir = './gt_png4'
output_dir = './gt_png3to1_channels'
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
filenames = os.listdir(root_dir)


for filenamei in filenames:
	if filenamei[-4:]=='.png':
		filenamei_path = os.path.join(root_dir,filenamei)
		imagei = cv2.imread(filenamei_path)
		grayi = cv2.cvtColor(imagei, cv2.COLOR_BGR2GRAY)

		cv2.imwrite(os.path.join(output_dir, filenamei),grayi)

