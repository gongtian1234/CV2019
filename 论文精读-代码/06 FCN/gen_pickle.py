import os

# 把文件处理成字典格式
modes = ['training', 'validation']
dirs = ['images', 'annotations']
root_dir = './Data_zoo2/MIT_ScenceParsing/ADEChallengeData2016/'

all_dict = {}

for mode in ['training', 'validation']:
    raw_images_dir = os.path.join(root_dir, 'images/', mode)
    annotas_dir = os.path.join(root_dir, 'annotations/', mode)
    print('当前原图路径为：',raw_images_dir)
    print('当前annotation路径为：',annotas_dir)
    raw_filenames = os.listdir(raw_images_dir)
    annotas_filenames = os.listdir(annotas_dir)
    print(raw_filenames)
    print(annotas_filenames)
    if len(raw_filenames)!=len(annotas_filenames):
        print('error: 数量不匹配')

    tmp_lists = []
    for raw_filename in raw_filenames:
        # 这一步是为了检查一下，原图和标注图是否对应
        if raw_filename.split('.')[0]+'.png' in annotas_filenames:
            pass
        else:
            print(0)
        tmp_dict = {}
        
        # 在写入路径前，先检查一下路径是否存在
        if os.path.exists(os.path.join(annotas_dir, raw_filename.split('.')[0]+'.png')):
            tmp_dict['annotation'] = os.path.join(annotas_dir, raw_filename.split('.')[0]+'.png')
        else:
            print(os.path.join(annotas_dir, raw_filename.split('.')[0]+'.png'),'路径不存在')
            
        tmp_dict['filename'] = raw_filename.split('.')[0]
        
        if os.path.exists(os.path.join(raw_images_dir, raw_filename)):
            tmp_dict['image'] = os.path.join(raw_images_dir, raw_filename)
        else:
            print(os.path.join(raw_images_dir, raw_filename), '路径不存在')
        
        tmp_lists.append(tmp_dict)
    all_dict[mode] = tmp_lists
    print()

with open('test.pickle', 'wb') as f:
    pickle.dump(all_dict, f, pickle.HIGHEST_PROTOCOL)
    
# 打开测试
# with open('test.pickle', 'rb') as f:
#     data = pickle.load(f)