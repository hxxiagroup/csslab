'''
简介：
    对数据的分布进行拟合
    即，概率密度函数符合哪种分布！！！ 并不是曲线拟合！！
        基于scipy中的stats和otimize两种方法


方法：
    分布拟合 - FitModel.fit - 基于scipy.stats的连续随机变量的拟合方法
    分布拟合 - FitModel.fit2 - 基于scipy.optimize.curvefit 的 频率密度直方图的数据进行曲线拟合
    绘制拟合 - FitMofel.plot_model


数据：
    data, 为pandas.Series数据类型
    data_pdf，pandas.Series数据类型，index为xdata,values为ydata(概率密度)
        也可以使用data_pdf来做一般性质的曲线拟合

备注：

    * 2017.10.27
        目前只是数据的分布进行拟合，后面可以再扩展成曲线的拟合较好
        支持的分布需要继续增加
    * 2017.10.30
        拓展曲线拟合，使用Fitmodel(data_pdf)，用data_pdf代替拟合的数据，就是曲线的拟合了。


    * 2017.11.1 - 重要
        修复严重错误，关于概率密度的计算，原来采用频率来作为概率
        目前改为np.histgram方法来计算密度
            第二中方法跟R语言中的density方法给出的结果一致，（也是张凌师兄的计算方法）

        第二，采用optimize.curfit方法得到的最优解，跟R语言中的fitdistplus包的结果差别较大

            发现scipy中有更好的曲线拟合方法，scipy.stats中的fit方法,好像数据过大，拟合时间较长
            接下来考虑用scipy.stats中分分布的拟合方法来做
                -存在的一个问题是，stats中所有的分布都采用统一的形式，对于分布的都采用loc,scale来控制，这可能也比较难理解
                -但是估计拟合的结果应该跟R中差不多


'''

import numpy as np

import pandas as pd
from scipy.special import gamma as _gamma
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt


class FitModel():

    DEFINED_DIST = ['powerlaw',
                    'exponential',
                    'gamma',
                    'lognormal',
                    'weibull',
                    'exponential_powerlaw']

    STATS_DIST = ['lognorm',
                  'expon',
                  'powerlaw',
                  'gamma',
                  'exponpow',
                  'norm',
                  'truncexpon',
                  'weibull_min',
                  'weibull_max'
                  ]

    INIT_PARA = {'powerlaw': [1,1.5],
                     'expon': [1,2],
                     'gamma': [1, 2],
                     'lognorm': [3, 2],
                     'weibull': [1, 2],
                     'exponpow':[5.3,1.5,0.9]}

    def __init__(self, data=None,data_pdf=None):
        '''
        :param data: 数据为pd.Series数据
        :param data_pdf: pd.Series, index为xdata，values为probability
        '''
        self.origin_data = data
        if data_pdf is None and data is not None:
            self.data_pdf = FitModel.distribution_pdf(data)
        else:
            self.data_pdf = data_pdf
        self.summary = []

    # ----------------------------------------------------------------------------
    @staticmethod
    def powerlaw(x, beta):
        return (beta-1) * (x ** (-beta))

    @staticmethod
    def lognorm(x, mu, sigmma):
        return 1 / (x * sigmma * np.sqrt(2 * np.pi)) * np.exp(((np.log(x) - mu) ** 2) / (-2 * sigmma * sigmma))

    @staticmethod
    def expon(x,lam):
        return lam * np.exp(-lam * x)

    @staticmethod
    def weibull(x, alpha, beta):
        '''对比一致'''
        return (alpha / beta) * ((x / beta) ** (alpha - 1)) * np.exp(-((x / beta) ** alpha))

    @staticmethod
    def gamma(x, alpha, beta):
        '''检测一致'''
        return ((beta ** alpha) / _gamma(alpha)) * (x ** (alpha - 1)) * np.exp(-beta * x)

    @staticmethod
    def exponpow(x, x0, beta, alpha):
        return (x + x0) ** (-beta) * np.exp(-alpha * x)

    # ----------------------------------------------------------------------------

    @staticmethod
    def distribution_absolute_pdf(data):
        '''
            计算数据的概率密度分布,最后的概率值加起来都等于1
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

    @staticmethod
    def distribution_pdf(data,bins=None):
        '''
        :param data:
        :return:
        '''
        if data is None:
            return None

        if bins is None:
            bins = 512
        density,xdata = np.histogram(data,bins=bins,density=True)
        xdata = (xdata + np.roll(xdata,-1))[:-1]/2.0
        data_pdf = pd.Series(density,index=xdata)
        return data_pdf

    def fit2(self, distribution, data=None, data_pdf=None, x_max=None, x_min=None, initial_para=None):
        '''
        对数据的概率密度分布进行拟合。
        拟合的信息会保存成Dict,包括:'distribution','popt', 'pcov', 'data_pdf','xdata','ydata'
        保存到模型的summary 中，以便查询和绘制结果

        :param distribution: 分布名称
        :param data: 需要分布拟合的数据
        :param data_pdf: 概率密度分布数据
        :param x_max: 拟合分布图像的上限
        :param x_min: 拟合分布图像的下限
        :param initial_para: 拟合分布初始参数
        :return: 拟合的信息，dict
        '''
        if initial_para is None:
            if distribution in FitModel.INIT_PARA.keys():
                initial_para = FitModel.INIT_PARA.get(distribution)
            else:
                print('- - 拟合的分布未定义 - - ')
                return None

        if data_pdf is None:
            if self.data_pdf is not None:
                #self.origin_data不是空的，那么self.data_pdf一定不是空
                data_pdf = self.data_pdf
            else:
                data_pdf = FitModel.distribution_pdf(data)

                if data_pdf is None:
                    print('Error: Data is None')
                    return None
                else:
                    self.data_pdf = data_pdf
        else:
            self.data_pdf = data_pdf

        if x_max is not None:
            data_pdf = data_pdf[data_pdf.index < x_max]

        if x_min is not None:
            data_pdf = data_pdf[data_pdf.index > x_min]

        xdata = np.asarray(data_pdf.index.values)
        ydata = np.asarray(data_pdf.values)


        fit_distribution = getattr(FitModel,distribution)


        popt, pcov = optimize.curve_fit(fit_distribution, xdata, ydata, p0=initial_para)
        fit_pdf = fit_distribution(xdata,*popt)

        sse = np.sum(np.power(data_pdf.values - fit_pdf, 2.0))

        fit_info = {'method':'FitModel',
                    'distribution': distribution,
                    'para': popt,
                    'pcov': pcov,
                    'sse':sse,
                    'data_pdf':data_pdf,
                    'xdata': xdata,
                    'ydata': ydata,
                    'fit_pdf':fit_pdf}
        self.summary.append(fit_info)

        print('------------ 拟合分布 %s -------------' % fit_distribution)
        print('- - Optimal Parameters : ',popt)
        print('-Estimated Covariance : ',pcov)
        return fit_info


    def fit(self,distribution,data=None,x_max=None,x_min=None):

        if distribution in FitModel.STATS_DIST:
            fit_distribution = getattr(stats,distribution)
        else:
            print('- - scipy.satas 未定义改分布 - - ')
            return None

        if data is None and self.origin_data is not None:
            data = self.origin_data

        if x_max is not None:
            data = data[data.index < x_max]

        if x_min is not None:
            data = data[data.index > x_min]

        # 开始拟合 --------------------
        print('------------ 拟合分布 %s -------------' % fit_distribution)
        para = fit_distribution.fit(data,floc=0)
        arg = para[:-2]
        loc = para[-2]
        scale = para[-1]
        data_pdf = FitModel.distribution_pdf(data)

        xdata = data_pdf.index.values
        ydata = data_pdf.values

        fit_pdf = fit_distribution.pdf(xdata,*arg,loc=loc,scale=scale)
        sse = np.sum(np.power(data_pdf.values - fit_pdf, 2.0))

        fit_info = {'method':'stats',
                    'distribution': distribution,
                    'para': para,
                    'pcov': sse,
                    'sse':sse,
                    'data_pdf': data_pdf,
                    'xdata': xdata,
                    'ydata': ydata,
                    'fit_pdf':fit_pdf}
        self.summary.append(fit_info)


    def plot_model(self,log_log=True,style=0, mfrow=None):
        '''
        :param log_log: 是否为log_log
        :param style: 绘制的风格
            0 - 所有拟合绘制在一起
            1 - 拟合结果分多个ax绘制
        :param mfrow: 图片分割方式，如果style为1的话，可以指定
        :return:
        '''
        if len(self.summary) < 1:
            print('- - 模型还没有拟合 - - ')
            return None

        fig = plt.figure()
        if style == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(self.data_pdf.index.values, self.data_pdf.values, 'k+')
            for model in self.summary:
                xdata = model.get('xdata')
                para = model.get('para')
                distribution = model.get('distribution')
                ydata = model.get('fit_pdf')
                ax.plot(xdata, ydata, label=distribution)
                ax.legend()
            ax.set_ylabel('Pr')
            if log_log:
                ax.set_yscale('log')
                ax.set_xscale('log')

        if style == 1:
            model_num = len(self.summary)
            if mfrow is None:
                # 设置图片分割方式，如果没有的话
                if model_num < 3:
                    mfrow = (1, model_num)
                elif model_num < 7:
                    mfrow = (2, 3)
                else:
                    mfrow = (3, 3)
            # 开始绘制每一个ax
            axes = []
            for i, model in enumerate(self.summary):
                axes.append(fig.add_subplot(mfrow[0], mfrow[1], i + 1))
                xdata = model.get('xdata')
                para = model.get('para')
                distribution = model.get('distribution')
                ydata = model.get('fit_pdf')

                fit_distribution = getattr(eval(model.get('method')),distribution)
                print(para)
                arg = para[:-2]
                loc = para[-2]
                scale = para[-1]
                ydata_2 = fit_distribution.pdf(xdata,*arg,loc=loc,scale=scale)

                axes[i].plot(self.data_pdf.index.values, self.data_pdf.values, 'k+')
                axes[i].plot(xdata, ydata, label=distribution)
                axes[i].plot(xdata,ydata_2,label='Fit')

                axes[i].legend()
                axes[i].set_ylabel('Pr')
                if log_log:
                    axes[i].set_yscale('log')
                    axes[i].set_xscale('log')

        plt.show()
        return fig

def test():
    xdata = np.linspace(0.1, 2.5, 100)
    lis_1 = [FitModel.weibull(x, 0.5, 1) for x in xdata]
    lis_2 = [FitModel.weibull(x, 1.5, 1) for x in xdata]
    lis_3 = [FitModel.weibull(x, 5, 1) for x in xdata]

    plt.plot(lis_1)
    plt.plot(lis_2)
    plt.plot(lis_3)
    plt.show()


def test_model():
    DataDir = r'G:\data'
    path_dis = DataDir + '\\DistanceAC.csv'
    data = pd.read_csv(path_dis, header=0)
    data = data['DistanceAC']
    data = data[data > 0]
    # -----------------------------------------------
    model = FitModel(data=data)
    model.fit(distribution='expon', x_max=8)
    model.fit(distribution='lognorm')
    model.fit(distribution='gamma')
    model.plot_model(style=1)

if __name__ == '__main__':
    test_model()



