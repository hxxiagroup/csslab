# -*- coding:utf-8 -*-
'''
目的：提供一些文件操作 和 数据处理的方法

方法：
    ----------功能 ------------ 方法 ---------------
    * 读取大的数据表(csv) - read_csv
    * 获取目录下所有某类型的文件名 - get_filename
    * 读取目录下所有某类型的数据（csv,xlsx） - connect_file
    * 数据表随机长度的抽样 - random_dataframe_sample
    * 数据表根据字段过滤 - dataframe_filter
    * 数据表根据时间戳过滤 - dataframe_slice_by_timestamp
    * 计算概率密度分布 - distribution
    * 计算累计概率密度分布 - distribution_cp
    * 数据归一化到某个区间 - normlize

备注：
    * 2017.10.16 - dataframe_filter方法还需要修改

'''

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def get_files(dir,filetype='.csv',complete_path=True):
    '''
    :param complete_path: 是否返回完整文件的path，否则返回文件的name(不包含拓展名)
    :return:
    '''
    #返回某个目录下所有后缀为指定类型的文件的list
    files = []
    for filename in os.listdir(dir):
        if os.path.splitext(filename)[1] == filetype:
            files.append(os.path.splitext(filename)[0])
    if complete_path:
        files = [os.path.join(dir,each+filetype) for each in files]
        return files
    else:
        return files


def get_subdir(sup_dir):
    '''放回某个目录下所有当前子目录'''
    DirList = []
    for subdir in os.listdir(sup_dir):
        abs_path = os.path.join(sup_dir,subdir)
        if os.path.isdir(abs_path):
            DirList.append(abs_path)
    return DirList

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

def dataframe_filter(data, attr, lower = None, upper = None):
    '''
    :param data:  pandas.Dataframe obeject.
    :param attr:  the attribution that you wanna choose for filt
    :param lower: the lower limit of data
    :param upper: the upper limit of data
    :return: pandas.Dataframe after filt
    '''
    origin_length = len(data)
    if lower is not None:
        data = data[data[attr] >= lower]
    if upper is not None :
        data = data[data[attr] <= upper]

    filt_length = len(data)
    left_rate = round(filt_length/origin_length,3)

    print('- - Filter: ',lower, ' ---- ', upper)
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

def distribution_fre(data):
    '''
        计算数据的频率密度分布,最后的概率值加起来都等于1
        :param data: list 或者 pandas.Series.
        :return: pandas.Series
        '''
    if data is None:
        return None
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    data_count = data.value_counts().sort_index()
    data_p = data_count / data_count.sum()
    return data_p

def distribution_pdf(data, bins=None):
    '''
    用频率密度直方图来估计概率密度分布
    :param data: 数据
    :return: data_pdf，pandas.Series
    '''
    if data is None:
        return None

    if bins is None:
        bins = 512
    density, xdata = np.histogram(data, bins=bins, density=True)
    xdata = (xdata + np.roll(xdata, -1))[:-1] / 2.0
    data_pdf = pd.Series(density, index=xdata)
    return data_pdf

def distribution_cdf(data, bins=None):
    pdf = distribution_pdf(data, bins)
    cdf = []
    for ind in pdf.index:
        cdf.append(pdf[ind:].sum())

    cdf = pd.Series(cdf, index=pdf.index)

    return cdf

def plot_distribution(data, subplot=2, data_norm=False, cmp=False, grid=True):
    '''
    :param data: Series数据
    :param subplot: 绘制原始的，log 和 log-log
    :param data_norm: 数据是否归一化，例如normlized degree
    :param cmp: 是否绘制累计密度概率
    :param grid: 网格线是否显示
    :return: None
    '''

    if data_norm:
        data_normed = normlize(data.values,0,1)
        name = 'Normlized'+ str(data.name)
        data = pd.Series(data_normed,name=name)

    ylabel = 'Probability'

    if cmp:
        data = distribution_cdf(data)
        ylabel = 'Cumulative ' + ylabel
    else:
        data = distribution_pdf(data)

    fg = plt.figure()
    ax1 = []
    for i in range(subplot):
        ax1.append(fg.add_subplot(1,subplot,i+1))

    data.plot(ax=ax1[0], style='*-')
    ax1[0].set_title('Distribution')

    if subplot>=2:
        data.plot(ax=ax1[1], style='*', logy=True, logx=True)
        ax1[1].set_title('log-log')
        #ax1[1].set_xlim([0, 50])

    if subplot>=3:
        data.plot(ax=ax1[2], style='*-', logy=True)
        ax1[2].set_title('semi-log')

    for i in range(subplot):
        ax1[i].set_ylabel(ylabel)
        ax1[i].set_xlabel(data.name)
        ax1[i].set_xlim([0, max(data.index)*1.1])
        ax1[i].grid(grid, alpha=0.8)

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
