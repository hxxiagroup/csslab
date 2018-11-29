
'''
#################################################
################## Colorfly #####################
#################################################

介绍：
    颜色工具,为数据设置合适的颜色，以达到更好的可视化效果

方法：
    hex2rgb()                           : 十六进制颜色转RGB颜色
    rbg2hex()                           : RGB颜色转十六进制颜色码

    classify_color_discrete(data,...)   : 为离散的数据设置颜色，支持传入颜色cmap
    classify_color_sequential(data,...) : 为连续的数据设置颜色，支持多种插值方式，自定义颜色层数

    get_cmap_colors(...)                : 获取离散cmap的颜色码

Reference：
    具体的颜色(cmap)可以参考下面网站
    - https://matplotlib.org/examples/color/colormaps_reference.html

Author: 文
QQ: 24585195
Date: 2018.11

'''

import numpy as np
import matplotlib.pyplot as plt

import colors_ as cs_


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

    :param rgb: list like,例如[255,255,255]
    :return: str, hex color
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

# --------------------------------------------------------

def _to_color(value, scale, maxvalue,minvalue,plt_cmap):
    '''
    
    根据为级别为value的值分配颜色，借鉴geoplotlib,
    其实就是将分级的level val映射到0-1之间,应该有更简单的办法，
        -比如_color_bar里面的映射方法
    '''
    if scale == 'lin':
        if minvalue >= maxvalue:
            raise Exception('minvalue must be less than maxvalue')
        else:
            value = 1. * (value - minvalue) / (maxvalue - minvalue)
    elif scale == 'log':
        if value < 1 or maxvalue <= 1:
            raise Exception('value and maxvalue must be >= 1')
        else:
            value = np.log(value) / np.log(maxvalue)
    else:
        raise Exception('scale must be "lin", "log", or "sqrt"')

    if value < 0:
        value = 0
    elif value > 1:
        value = 1
    # value = int(1. * level * value) * 1. / (level - 1)
    camp_cols = plt_cmap(value)
    color = [int(c * 255) for c in camp_cols[:3]]
    return color, value

def _plot_discrete_colorbar(color_dict):
    xlabels = list(zip(*color_dict))[0]
    fig,ax = plt.subplots(1,1)
    c_num = len(color_dict)

    x_s = 0.1
    x_e = 0.9
    p_wd = (x_e - x_s) / (c_num + 1)
    locs = [p_wd * (i+1) for i in range(c_num + 1)]
    xticks_loc = [(locs[i]+locs[i+1])/2 for i in range(c_num)]

    for i,each in enumerate(color_dict):
        color_i = each[1]
        ax.axvspan(xmin=locs[i],xmax=locs[i+1],color=color_i)
        
    ax.set_xticks(xticks_loc) 
    ax.set_xticklabels(xlabels)
    ax.set_xlim([0,1])

    plt.show()
def _plot_sequential_colorbar(vals, cmap):
    '''绘制colorbar'''
    num_val = len(vals)
    num_gradient = 50
    step = num_gradient / num_val

    gradient = np.linspace(1, 0, num_gradient)
    gradient = np.transpose([gradient, gradient])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ytick = [i * step for i in range(0, num_val, 1)]
    ylabel = [str(round(each, 2)) for each in vals[::-1]]

    ax.set_yticks(ytick)
    ax.set_yticklabels(ylabel)
    ax.get_xaxis().set_visible(False)
    ax.set_title(cmap)
    ax.imshow(gradient, cmap=cmap)

    plt.tight_layout()
    plt.show()

# --------------------------------------------------------

def classify_color_discrete(data,cmap=None,hexcolor=True,
        random_color=False,show_colorbar=False,need_return_levels=False):
    '''
    Introduction:
        离散数据的颜色选择工具
    Params:
        data: 
            list like, 例如，list,tuple,pd.Series
        cmap: 
            str or list of str, 表示离散颜色系列的名称，例如"Set1",["Paired","Set1"],
            默认会自动选择颜色
        hexcolor: 
            bool, 表示是否使用十六进制颜色
        show_colorbar:
            bool, 展示各组颜色的结果
        random_color:
            bool, 表示是否随机选择颜色，否则按cmap的顺序选取
        need_return_levels: 
            bool,表示是否需要返回颜色的字典，格式如{"张三":"#FFFFF",}

    Returns:
        color_ret:
            list,表示各个数据对应的颜色，跟输入的数据一样的长度
            例如，["#12345F","#23456F",....]

        color_levels:
            list,如果设置need_return_levels为True
            格式如[[0,10,"#433A83"],[10,20,"FFFFF"],...]，
            表示数据0 - 10之间是一组，颜色是#433A83
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
    color_levels = list(color_dict.items())

    if(show_colorbar):
        _plot_discrete_colorbar(color_levels)

    if(need_return_levels):
        return color_ret,list(color_levels)
    else:
        return color_ret


def classify_color_sequential(data,scale="lin",logbase=10,cmap="viridis",
        level=10,hexcolor=True,need_return_levels=False,show_colorbar=False):
    '''
    Params:
        data: 
            list like, 例如，list,tuple,pd.Series
        scale: 
            str, 数据插值方式,["lin","log"]
        logbase: 
            int,log插值的底数，默认为10为底
        cmap: 
            str or list of str, 表示离散颜色系列的名称，例如"Set1",["Paired","Set1"]
        level: 
            int,指定数据分组数量，分组越多，颜色越多,默认分组10
        hexcolor: 
            bool, 表示是否使用十六进制颜色
        need_return_levels: 
            bool,表示是否需要返回颜色的分组信息，

    Returns:
        color_ret:
            list,表示各个数据对应的颜色，跟输入的数据一样的长度
            例如，["#12345F","#23456F",....]

        color_levels:
            list,如果设置need_return_levels为True
            格式如[[0,10,"#433A83"],[10,20,"FFFFF"],...]，
            表示数据0 - 10之间是一组，颜色是#433A83
    '''
    from matplotlib.pylab import get_cmap

    data_ = np.asarray(data)
    

    if(scale not in SUPPORT_SCALES):
        error_info = "Error: 不支持的插值类型，选择下面的插值方式：{}".format(str(SUPPORT_SCALES))
        raise Exception(error_info)

    # -------------------------先写到这啦
    try:
        plt_cmap = get_cmap(cmap)
    except ValueError as e:
        print(e)
        print('pylab中没有找到该颜色: {}'.format(cmap))
        raise ValueError

    # -----------插值分组-----------------
    vmax = data_.max()
    vmin = data_.min()

    if scale == 'lin':
        level_values = np.linspace(vmin, vmax, level + 1)
    elif scale == "log":
        vmin_log = np.log10(vmin) / np.log10(logbase)
        vmax_log = np.log10(vmax) / np.log10(logbase)
        level_values = np.logspace(vmin_log, vmax_log, level + 1,base=logbase)
    else:
        return None

    # 颜色插值分组
    colors_res = np.asarray([None for i in range(len(data_))])
    return_dic = []
    #  color_vals = []
    print('-----------Levels:{}------------'.format(level))
    for i in range(1, len(level_values), 1):
        # 取上不取下，最小的级别取下
        upper = level_values[i]
        lower = level_values[i - 1]

        # 取改组中间值作为平均颜色，
        mean_val = (level_values[i - 1] + level_values[i]) / 2
        color_i, color_val = _to_color(mean_val, scale, vmax,vmin,plt_cmap)

        if hexcolor:
            color_i = rgb2hex(color_i)

        if i == 0:
            idx = np.where((data_>=lower) & (data_<=upper)) 
        else:
            idx = np.where((data_>lower) & (data_<=upper)) 

        colors_res[idx] = color_i

        return_dic.append([lower,upper,color_i])
        #  color_vals.append(color_val)
        print('Level: ({:^10.3f}  -  {:^10.3f}], Color: {:^16}'.format(lower, upper, str(color_i)))

    # 处理空值
    if hexcolor:
        colors_res[np.where(colors_res==None)] = rgb2hex([0, 0, 0])
    else:
        colors_res[np.where(colors_res==None)] = [0, 0, 0]

    if show_colorbar:
        _plot_sequential_colorbar(level_values, cmap)

    if(need_return_levels):
        return colors_res,return_dic

    else:
        return colors_res



def test_discrete_colors():
    data = ["class_1","class_2","class_3","class_4","class_5","class_6"]
    colors,color_dict = classify_color_discrete(data,show_colorbar=True,need_return_levels=True)
    print(colors)
    print(color_dict)


def test_sequetial_colors():
    import random
    data = [i for i in range(2,300,1) if random.random()>0.3]

    ret,vals = classify_color_sequential(data,scale="log",show_colorbar=True,need_return_levels=True)
    print(vals)

    classify_color_sequential(data,scale="lin",show_colorbar=True)




if __name__ == "__main__":

    test_discrete_colors()

    test_sequetial_colors()





