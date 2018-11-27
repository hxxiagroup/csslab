
'''
介绍：
    颜色工具

方法：

    hex2rgb： 十六进制颜色转RGB颜色
    rbg2hex： RGB颜色转十六进制颜色码

    classify_color_discrete(...)： 为离散的数据设置颜色，支持传入颜色cmap
    classify_color_sequential(...): 为连续的数据设置颜色，支持多种插值方式，自定义颜色层数

    get_cmap_colors(...): 获取cmap的颜色码

Reference：
    具体的颜色(cmap)可以参考下面网站
    - https://matplotlib.org/examples/color/colormaps_reference.html

'''

import numpy as np
import colors_ as cs_

import pandas as pd
import matplotlib.pyplot as plt


def hex2rgb(hexcolor):
    '''
    把十六进制颜色转化为RGB,
    如 #6F1958,返回[111, 25, 88]

    :param hexcolor: 以#开头，例如: #6F1958
    :return: list
    '''
    hexcolor = hexcolor[1:] if '#' in hexcolor else hexcolor[:]
    hexcolor = int('0x' + hexcolor, base=16)

    rgb = [(hexcolor >> 16) & 0xff,
           (hexcolor >> 8) & 0xff,
           hexcolor & 0xff]
    return rgb


def rgb2hex(rgb):
    '''
    把RGB颜色转化为十六进制颜色
    例如：[111, 25, 88] 返回 #6F1958

    :param rgb: list
    :return: str, hex
    '''
    hexcolor = '#'
    for each in rgb:
        hex_each = hex(each)[2:].upper()
        if len(hex_each) < 2:
            hex_each += '0'
        hexcolor += hex_each
    return hexcolor


MAX_DISCRETE_NUM = cs_.common_discrete_color_num

SUPPORT_SCALES = ("lin","log",)

def random_colors(color_num):
    pass

def get_cmap_colors(cmap):
    return cs_.get_colors(cmap)


def classify_color_discrete(data,cmap=None,hexcolor=True,
        random_color=False,need_return_dict=False):
    '''
    Introduction:
        离散数据的颜色选择工具

    Params:
        data: 
            list like, 例如，list,tuple,pd.Series
        cmap: 
            str or list of str, 表示离散颜色系列的名称，例如"Set1",["Paired","Set1"]
        hexcolor: 
            bool, 表示是否使用十六进制颜色
        random_color:
            bool, 表示是否随机选择颜色，否则按cmap的顺序选取
        need_return_dict: 
            bool,表示是否需要返回颜色的字典，格式如{"张三":"#FFFFF",}

    Returns:
        color_ret:
            list,颜色的列表，跟输入的数据一样的长度
        color_dict:
            dict,如果设置need_return_dict为True
    '''
    if cmap is not None:
        if(isinstance(cmap,str)):
            cmap = (cmap,)
        unsupport_cmaps = cs_.find_unsupport_cmaps(cmap)
        if(unsupport_cmaps):
            print("存在不支持的颜色：",*unsupport_cmaps)
            raise ValueError
        
        access_colors = cs_.get_colors(cmap)
    else:
        access_colors = cs_.get_common_discrete_colors()


    data = np.asarray(data)
    data_unique_ = set(data)
    num_unique_ = len(data_unique_)

    if(num_unique_ > len(access_colors)):
        print('''Warning: 离散值种类数( {} )超过可用颜色数( {} )'''
                .format(num_unique_,len(access_colors)))
        print("使用cmap增加更多的离散颜色系列吧！")

    color_use = access_colors[:num_unique_].copy()
    if(random_color):
        import random
        random.shuffle(color_use)

    color_dict = {}
    for val_,color_ in zip(data_unique_,color_use):
        if(hexcolor):
            color_dict[val_] = color_
        else:
            color_dict[val_] = hex2rgb(color_)

    color_ret = list(map(color_dict.get,data))

    if(need_return_dict):
        return color_ret,color_dict
    else:
        return color_ret


def classify_color_sequential(data,scale="lin",cmap="viridis",
        level=10,hexcolor=True):

    if isinstance(data, list) or isinstance(data, tuple):
        data_by = np.asarray(data)

    
    if(scale not in SUPPORT_SCALES):
        print("Error: 不支持的插值类型，选择下面的插值方式：",*SUPPORT_SCALES)
        raise ValueError

    
    # -------------------------先写到这啦


