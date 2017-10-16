# -*- coding:utf-8 -*-
'''

目的：提供一些文件操作 和 数据处理的方法

方法一览：
    功能 - 方法

    * 读取大的数据表(csv) - read_csv
    * 获取目录下所有某类型的文件名 - get_filename
    * 读取目录下所有某类型的数据（csv,xlsx） - connect_file
    * 数据表随机长度的抽样 - random_dataframe_sample
    * 数据表根据字段过滤 - dataframe_filter
        - 还需要修改
    * 数据表根据时间戳过滤 - dataframe_slice_by_timestamp
    * 计算概率密度分布 - distribution
    * 计算累计概率密度分布 - distribution_cp
    * 数据归一化到某个区间 - normlize


'''

import os
import random
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_csv(readpath):
    '''
    分块读取大的csv
    :param readpath: filepath
    :return: pd.DataFrame
    '''
    print(' - - start to read - - %s'%readpath)
    reader = pd.read_csv(readpath,iterator=True)
    loop = True
    chunkSize = 100000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
    read_d = pd.concat(chunks,ignore_index=True)
    return read_d

def get_filename(dir, filetype='.csv'):
    #返回某个目录下所有后缀为指定类型的文件名的list，不包含后缀
    filelist = []
    for filename in os.listdir(dir):
        if os.path.splitext(filename)[1] == filetype:
            filelist.append(os.path.splitext(filename)[0])
    return filelist

def connect_file(dir, connect_nums=None, filetype='.csv',header = 0):
    '''
    连接某一个目录下的所有数据，可以指定文件数量
    :param dir: the directory of files
    :param connect_nums:  需要连接的文件序号，如果为空，则默认为全部文件,例如第1，2，5个。[0,1,4]
    :param filetype: the type of files you want to get all data from
    :return: Dataframe of all data
    '''
    filelist = get_filename(dir, filetype=filetype)
    all_data = []

    if connect_nums is None:
        connect_nums = range(len(filelist))
    if dir[-1] != '\\':
        dir = dir + '\\'
    #序号超出文件数量，还有其他很多情况，学学try的用法吧，2017.6.12
    elif max(connect_nums) > len(filelist) - 1:
        print('- - Mistake - - Connect num is wrong')
        return None

    if len(filelist) != 0:
        for file_id in connect_nums:
            readpath = dir + filelist[file_id] + filetype
            #注意有的文件的encoding类型
            print(' -- connecting file : %s '%readpath)
            each_data = pd.read_csv(readpath, header=header)
            all_data.append(each_data)
    else:
        print('- - - - - No file found - - - - - ')
        return None
    return pd.concat(all_data,ignore_index=True)

def random_dataframe_sample(df, num_sample):
    '''
    :param df: DataFrame，数据表
    :param num_sample: 随机抽样数量
    :return: DataFrame，抽取的样本数据表
    '''
    inds = list(df.index)
    if num_sample <= len(df):
        ind_sample = random.sample(inds, num_sample)
        df_sample = df.ix[ind_sample, :]
    else:
        df_sample = df
    return df_sample

def dataframe_filter(data, attr, lower_limit = 'unlimited', upper_limit = 'unlimited'):
    '''
    :param data:  pandas.Dataframe obeject.
    :param attr:  the attribution that you wanna choose for filt
    :param lower_limit: the lower limit of data
    :param upper_limit: the upper limit of data
    :return: pandas.Dataframe after filt
    '''
    origin_length = len(data)
    #----------------------------------
    if upper_limit == 'unlimited':
        pass
    elif isinstance(eval(upper_limit),float) or isinstance(eval(upper_limit),int):
        data = data[data[attr] <= eval(upper_limit)]
    else:
        print('- - Mistake: upper_limit should be a string of digit')
        return None
    #-----------------------------------
    if lower_limit =='unlimited':
        pass
    elif isinstance(eval(lower_limit),float) or isinstance(eval(lower_limit),int):
        data = data[data[attr] >= eval(lower_limit)]
    else:
        print('- - Mistake: lower_limit should be a string of digit')
        return None

    filt_length = len(data)
    left_rate = round(filt_length/origin_length,3)

    print('- - Filter:  %s  --  %s'%(lower_limit,upper_limit))
    print('- - orgin data: %d  - - after filtering: %d  - - %.3f left'
              %(origin_length,filt_length,left_rate))

    return data

def dataframe_slice_by_timestamp(df,filter_dict):
    '''
    根据时间戳截取DataFrame数据表
    :param df: DataFrame
    :param filter_dict: dict, e.g. {'Timestamp':['2008-10-1 10;00:00','2008-10-1 12:00:00'],}
    :return: DataFrame slice
    '''
    ind_all = []
    for attr,attr_vlaue in filter_dict.items():
        timedata = df[[attr]].copy()
        timedata['IndexValue'] = list(timedata.index.values)
        timedata.index = list(timedata[attr].values)
        timedata = timedata[attr_vlaue[0]:attr_vlaue[1]]
        ind_got = timedata['IndexValue']
        ind_all.append(list(ind_got.values))
    #get intersection of those index
    if len(ind_all) > 1:
        set_ind1 = set(ind_all[0])
        ind_return = list(set_ind1.intersection(*ind_all[1:]))
    else:
        ind_return = ind_all[0]
    df = df.ix[ind_return,:]
    return df

def distribution(sd):
    '''
    计算数据的概率密度分布
    :param sd: list 或者 pandas.Series.
    :return: pandas.Series
    '''
    if isinstance(sd,pd.Series):
        pass
    else:
        sd = pd.Series(sd)
    sd_count = sd.value_counts().sort_index()
    sd_fre = sd_count/sd_count.sum()
    return sd_fre

def distribution_cp(sd):
    '''
    :purpose: calculate the cumulative probability distribution of a group data.
    :param sd: list of pandas's Series.
    :return: pandas's Series
    '''
    sd = pd.Series(sd)
    sd_count = sd.value_counts().sort_index()
    sd_fre = sd_count/sd_count.sum()
    origin_index = sd_fre.index.values
    sd_fre.index = range(len(sd_fre))
    sd_cp = sd_fre.copy()
    #分布函数，X<x
    for i in range(len(sd_fre)):
        sd_cp[i] = sum(sd_fre[:i])
    sd_cp.index = origin_index
    return sd_cp

def normlize(data,lower=0,upper=1):
    '''
    将数据规范化到某个区间
    :param data: 可以是list，array, ndarray等
    :param lower: 规范化的下界
    :param upper: 规范化的上界
    :return: 规范化的数据
    '''
    xmax = np.max(data)
    xmin = np.min(data)
    data_new = (upper - lower) * (data - xmin) / (xmax - xmin) + lower
    return data_new
