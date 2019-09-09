from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random, sys

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def random_captcha_text(char_set=number, captcha_size=4):
    # 验证码列表
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

def get_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()    # 返回的是一个列表，如['3', '4', '5', '6']
    captcha_text = ''.join(captcha_text)    # 将列表拼接成一个字符串
    # 生成验证码
    captcha = image.generate(captcha_text)
    image.write(captcha_text, 'captcha/images/'+captcha_text+'.jpg')


if __name__=='__main__':
    num = 6000
    for i in range(num):
        get_captcha_text_and_image()
        sys.stdout.write('\r>>Creating image %d, %d'%(i+1, num))    # sys.stdout.write(str) 输出的另一种方式，这种输出不会自动换行
        sys.stdout.flush()    # 保证一秒输出一个结果，要不然上面的结果等程序执行完成后才会输出
    sys.stdout.write('\n')
    sys.stdout.flush()
    print('生成完毕')










