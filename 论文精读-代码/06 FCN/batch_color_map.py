# -*- coding:utf-8 -*-

import argparse
import glob as gb
import numpy as np
from PIL import Image


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ ( np.uint8(str_id[-1]) << (7-j))
            g = g ^ ( np.uint8(str_id[-2]) << (7-j))
            b = b ^ ( np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap



def main():
    '''
    batch convert signle channel labels to 
    RGB color images
    
    src_path: the file path for signle channel labels
    target_path: the file path for rgb labels out
    num: the num for classify
    
    e.g.
        python batch_color_map.py labels out 3
    format:
        python batch_color_map.py [src_path] [target_path] [num]
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('src_path')
    parser.add_argument('target_path')
    parser.add_argument('num', type = int)
    args = parser.parse_args()

    src_path = args.src_path
    target_path = args.target_path
    num = args.num

    cmap = labelcolormap(num)
    pattern = '*.png'

    for file in gb.glob(src_path +'/'+ pattern):
        file_name = file.split('/')[-1].split('.')[0]
        print("Processing..." + file_name)

        im = Image.open(file)
        im_rgb = im.convert('RGB')

        imsz = im.size
        im = np.array(im)
        im_rgb = np.array(im_rgb)

        for i in range(0,imsz[1]):
            for j in range(0, imsz[0]):
                im_rgb[i,j] = cmap[im[i,j]]
        
        im_rgb = Image.fromarray(im_rgb)
        im_rgb.save(target_path + '\\' +  file_name.split('\\')[1] + '.png')

    print('done...')


if __name__ == '__main__':
    
    main()