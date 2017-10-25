'''
目的：
    截图工具
    设定截图的初始位置，截图大小，延时截图

备注：
    如果在主程序中加入循环，就可以实现连续的截图工作了！
    这完全是现有软件没找到这个功能，为了偷懒才写的！！
'''

from PIL import Image, ImageGrab
import time
import os


def screenshot(location,image_size,save_path,delay_time=5):
    '''
    :param location: 图片右上角位置，对应显示器中的位置（这个好像一般不容易知道），可能需要借助工具
    :param image_size: 图片大小
    :param save_path: 图片保存地址
    :param delay_time: 延迟时间
    :return: None
    '''
    time.sleep(delay_time)
    box = (location[0], location[1], location[0] + image_size[0], location[1] + image_size[1])
    im = ImageGrab.grab(box)
    im.save(save_path)
    print('成功截图:',save_path)
    return im


def example():
    Location = (655, 123)
    image_size = ((755, 755))
    while True:
        name_num = input('输入保存文件号码(输入 shut 结束)：\n')
        if name_num == 'shut':
            break
        file_name = 'screenshot' + str(name_num) + '.png'
        FILE_DIR = r'G:\data'
        save_path = os.path.join(FILE_DIR, file_name)
        screenshot(location=Location,image_size=image_size,save_path=save_path)

