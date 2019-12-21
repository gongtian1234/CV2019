#!/usr/bin/python

import json
import os.path as osp
import glob as gb
import PIL.Image
import argparse
import numpy as np

import sys

sys.path.append('../')
from labelme import utils


def our_labelme_shapes_to_label(img_shape, shapes):
    # label_name_to_val = {'background': 0, 'pool':1, 'closestool':2,'window':3,'curtain':4}
    label_name_to_val = {'background': 0, 'bowl':1, 'gua':2}
    lbl = np.zeros(img_shape[:2], dtype=np.int32)
    for shape in sorted(shapes, key=lambda x: x['label']):
        polygons = shape['points']
        label_name = shape['label']
        if label_name in label_name_to_val:
            label_value = label_name_to_val[label_name]
        else:
            label_value = int(label_name) 
            label_name_to_val[label_name] = label_value
            # label_value = len(label_name_to_val)
            # label_name_to_val[label_name] = label_value
        mask = utils.polygons_to_mask(img_shape[:2], polygons)
        lbl[mask] = label_value

    lbl_names = [None] * (max(label_name_to_val.values()) + 1)
    for label_name, label_value in label_name_to_val.items():
        lbl_names[label_value] = label_name

    return lbl, lbl_names


def main():
    '''
    batch convert json to signle channel labels 
    src_path: the file path for json file
    target_path: the file path for labels out
    format:
        python batch_json_to_dataset.py [src_path] [target_path]
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('src_path')
    parser.add_argument('target_path')
    args = parser.parse_args()

    src_path = args.src_path
    target_path = args.target_path
    pattern = '*.json'

    for jpath in gb.glob(src_path + "/" + pattern):
        json_name = jpath.split('/')[-1].split('.')[0]
        data = json.load(open(jpath))
        img = utils.img_b64_to_arr(data['imageData'])
        lbl, lbl_names = our_labelme_shapes_to_label(img.shape, data['shapes'])
        print(json_name.split('\\')[1])
        print(osp.join(target_path, json_name+'.png'))
        PIL.Image.fromarray(lbl).save(osp.join(target_path, json_name.split('\\')[1] + '.png'))

    print('done...')


if __name__ == '__main__':
    main()
