import os

# 此文件用于初始化参数


#
# path and dataset parameter
#

# DATA_PATH = 'data'  # 原来
DATA_PATH = 'data1'

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')

# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')
WEIGHTS_FILE = None

# CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#            'train', 'tvmonitor']   # 原来

CLASSES = ['bowl']

FLIPPED = True

#
# model parameter
#

IMAGE_SIZE = 448

CELL_SIZE = 7

BOXES_PER_CELL = 2

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0

ALPHA = 0.1

DISP_CONSOLE = False

#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.0001

# BATCH_SIZE = 45   # 原来
BATCH_SIZE = 1

# MAX_ITER = 15000
MAX_ITER = 3000

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

# SUMMARY_ITER = 10
SUMMARY_ITER = 1

# SAVE_ITER = 1000
SAVE_ITER = 100

#
# test parameter
#

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5
