
# coding: utf-8

# # Matplotlib 基础教程

# ### Matplotlib官方gallery：
# ### http://matplotlib.org/1.4.3/gallery.html
# ### 一个很好的教程：
# ### https://liam0205.me/2014/09/11/matplotlib-tutorial-zh-cn/
# ### GitHub:
# ### https://github.com/RiptideBo/csslab
# 
# ### 说明：
#     本教程是介绍我使用过的matplotlib中所有方法的汇总，方便大家在制图时查找使用。
#     我尽可能详细的备注了每个方法以及方法中参数的含义。
#     之前我画过的所有图也会在百度云第二组我个人的文件夹中上传一份。
#     
# ### @ author: Qian Pan
# ### @ E-mail: lovelyrita@mail.dlut.edu.cn

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # 功能更加强大的子图绘制方法
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes  # 用于插入图中的小图框
from matplotlib.ticker import MultipleLocator  # 设置坐标轴刻度
from mpl_toolkits.axes_grid1 import host_subplot  # 让图的坐标轴看起来更精美的方法
import mpl_toolkits.axisartist as AA  # 让图的坐标轴刻度向内
from collections import OrderedDict


# In[2]:

# 自己定义的一些线条类型
def get_linestyles(key):
    linestyles = OrderedDict(
        [('solid',               (0,())),
         ('loosely dotted',      (0,(1,10))),
         ('dotted',              (0, (1, 5))),
         ('densely dotted',      (0, (1, 1))),

         ('loosely dashed',      (0, (5, 10))),
         ('dashed',              (0, (5, 5))),
         ('densely dashed',      (0, (5, 1))),

         ('loosely dashdotted',  (0, (3, 10, 1, 10))),
         ('dashdotted',          (0, (3, 5, 1, 5))),
         ('densely dashdotted',  (0, (3, 1, 1, 1))),

         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
         ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
    return linestyles[key]


# ## 基本的点线图

# In[3]:

# 这个图是很多基础方法的汇总，所以没能照顾到可读性，大家看看方法好了
# 创建一个8 * 6点(point)的图,dpi设置图片分辨率，默认为80,facecolor的背景色
fig = plt.figure(num=1,figsize=(8,8),dpi=80,facecolor=None)

# 创建一个新的子图，axes_class用于创建坐标轴类型
ax1 = host_subplot(111,axes_class = AA.Axes)

x = np.linspace(-np.pi, np.pi, 256,endpoint=True)
y1,y2 = np.cos(x), np.sin(x)

# 移动脊柱：移动坐标轴到图中间，需要将其中的两条设置为无色
# 然后调整剩下的两条到合适的位置————数据空间的0点
#==============================================================================
# ax1 = plt.gca()
# ax1.spines['right'].set_color('none')
# ax1.spines['top'].set_color('none')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.spines['bottom'].set_position(('data',0.25))
# ax1.yaxis.set_ticks_position('left')
# ax1.spines['left'].set_position(('data',0.25))
#==============================================================================

ax1.plot(x,y1,'o', alpha=0.5,ms=5,mfc='y',mec='r',mew=0.5,label='line 1')
ax1.plot(x,y2,c = 'r', alpha=0.8,lw=2.5,ls = get_linestyles('loosely dashed'),label='line 2')

# 可以在坐标中中插入子坐标系
#ax2 = fig.add_axes([0.6,0.2,0.2,0.2])  # add_axes(rect[left,bottom,width,height]) 
#ax2.plot([0.1,0.2,0.3,0.4],[0.1,0.2,0.3,0.35],'bo',ms=2.5)
'''
c: color
alpha: 设置透明度
'bo': 表示画点图，点的颜色为'b'蓝色
ms: markersize: 表示点的大小，如果画线图则是linewidth
mfc: markerfacecolor: marker圆的颜色
mec: markeredgecolor: marker边的颜色
mew: markeredgewidth: marker圆的粗细
ls: linestyle: 线条的类型
lw: linewidth: 线条的粗细
label: 线条的标签，详细的设置用plt.legend
add_axes: 可以通过add_axes在图中添加子图，设置方形图框的四点坐标即可
'''
# 精细化设置
# 在图中添加文本框
ax1.text(-3, 0.53, r"$R^2=0.997$",fontsize=10)
ax1.text(-3, 0.43, r"$p\_value=8.94E^{-117}$",fontsize=10)

# 添加直线和标记点
t = 2*np.pi/3
ax1.plot([t,t],[0,np.cos(t)], color ='blue', linewidth=2, linestyle="--")
ax1.scatter([t,],[np.cos(t),],30, color ='blue')

#添加箭头
ann_first = ax1.annotate(r"$y=1.03e^{-0.032x}$", xy=(-1.5, 0), #xy箭头的位置
             xytext=(0, 0.35),fontsize=10,   # xytext文本框的位置
             arrowprops=dict(arrowstyle = '<-',linestyle = '-',
                             connectionstyle="arc3,rad=.2"))

# 加入小插图--放大大图中的一段图形
axins = zoomed_inset_axes(ax1, 0.5, loc=6)
axins.set_aspect('equal')
axins.plot(x[30:200],y1[30:200], 'yo', alpha=0.25, ms=2)
axins.plot(x[30:200],y2[30:200], 'r--',alpha=0.5,linewidth = 1.8)
plt.xticks(visible=True,fontsize = 8)
plt.yticks(visible=True,fontsize = 8)

# 加入小插图2--先插入一个子坐标，然后再画点
axins1 = inset_axes(ax1,
                   width="40%",  # width = 30% of parent_bbox
                   height=1.,  # height : 1 inch
                   loc=8)

axins1.plot(x[50:250],y1[50:250], 'yo', alpha=0.25, ms=2)
axins1.plot(x,y2, 'r--',alpha=0.5, linewidth = 1.8)
axins1.set_xticks([])
axins1.set_yticks([])

ax1.set_title('Testing',fontsize = 15)  # 图名称
ax1.set_xlabel('X')  # 纵横坐标轴label
ax1.set_ylabel('Ylabel')
ax1.set_xlim(min(x)-0.1,max(x+0.1))  # 设置坐标轴范围
ax1.set_ylim(min(y1)-0.1,max(y1+0.1))  # 设置坐标轴范围
# 更好的方式
#==============================================================================
# xmin, xmax = x.min(),x.max()
# ymin, ymax = y1.min(),y1.max()
# dx = (xmax - xmin) * 0.1
# dy = (ymax - ymin) * 0.1
# ax1.set_xlim(xmin-dx,xmax+dx)
# ax1.set_ylim(ymin-dx,ymax+dy)
#==============================================================================

ax1.set_xticks(np.linspace(-4,4,9,endpoint=True))  # 设置x周刻度
ax1.axis["left"].label.set_fontstyle('italic')  # 设置label字体为斜体
ax1.axis["bottom"].label.set_fontsize(14)  # 设置label字号
ax1.axis["left"].label.set_fontsize(14)
ax1.axis['left'].major_ticklabels.set_fontsize(12)  # majorticklabel字号
ax1.axis['bottom'].major_ticklabels.set_fontsize(12)

# 坐标轴上的记号标记被挡住时：可以放大，添加半透明底框
#for label in ax1.get_xticklabels() + ax1.get_yticklabels():
#    label.set_fontsize(16)
#    label.set_bbox(dict(facecolor='white',ec='None',alpha=0.65))

ax1.legend(loc=2)
# 以分辨率72来保存图片
# savefig("test.png",dpi=72)
plt.show()


# ## 子图--可以使用subplot,也可以使用gridspec功能更加强大

# In[4]:

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

G = gridspec.GridSpec(3, 3)  # 新建一个3行3列的图

axes_1 = plt.subplot(G[0, :])  # 第0个子图
plt.xticks([]), plt.yticks([])  # 坐标轴无刻度
axes_1.text(0.5,0.5, 'Axes 1',ha='center',va='center',size=24,alpha=.5)  # 字体左右水平居中，设置字号及透明度

axes_2 = plt.subplot(G[1,:-1])
plt.xticks([]), plt.yticks([])
axes_2.text(0.5,0.5, 'Axes 2',ha='center',va='center',size=24,alpha=.5)

axes_3 = plt.subplot(G[1:, -1])
plt.xticks([]), plt.yticks([])
axes_3.text(0.5,0.5, 'Axes 3',ha='center',va='center',size=24,alpha=.5)

axes_4 = plt.subplot(G[-1,0])
plt.xticks([]), plt.yticks([])
axes_4.text(0.5,0.5, 'Axes 4',ha='center',va='center',size=24,alpha=.5)

axes_5 = plt.subplot(G[-1,-2])
plt.xticks([]), plt.yticks([])
axes_5.text(0.5,0.5, 'Axes 5',ha='center',va='center',size=24,alpha=.5)

plt.show()


# ## 散点图--scatter或plot

# In[5]:

import matplotlib.pyplot as plt
import numpy as np
N = 50
x1 = np.random.rand(N)
y1 = np.random.rand(N)
x2 = np.random.rand(N)
y2 = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses
plt.scatter(x1, y1, s=area, c='lightsalmon', edgecolor='k',alpha=0.5)
plt.scatter(x2, y2, s=area, c='yellow', edgecolor='k',alpha=0.5)
plt.show()


# ## 柱状图--bar或者hist等 

# In[6]:

import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1*height, '%.2f'%float(height),
                ha='center', va='bottom')
n = 12
X = np.arange(n)
Y1 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)
Y2 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)

rects1 = plt.bar(X, +Y1, facecolor='pink', edgecolor='white')
rects2 = plt.bar(X, -Y2, facecolor='#fff000', edgecolor='white')  # 颜色不仅可以通过颜色名称，也可以通过rgb色号

autolabel(rects1)  # 在柱状图上添加label

plt.ylim(-1.25,+1.25)
plt.show()


# ## 等高线图--contourf 

# In[7]:

import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X,Y = np.meshgrid(x,y)

plt.axes([0.025,0.025,0.95,0.95])

plt.contourf(X, Y, f(X,Y), 8, alpha=.75, cmap=plt.cm.hot)  # 绘制色块
C = plt.contour(X, Y, f(X,Y), 8, colors='black', linewidth=.5)  # 等高线
plt.clabel(C, inline=1, fontsize=10)  # 等高线上的label

plt.xticks(visible=False)  # 这两种表达方式都是tick不可见，但下面的效果更好
plt.yticks([])
plt.show()


# ## 灰度图或者热图--imshow

# In[8]:

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

image = np.random.uniform(size=(10, 10))
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax.set_title('dropped spines')

# Move left and bottom spines outward by 10 points
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.show()


# ## 饼图--pie 

# In[9]:

import matplotlib.pyplot as plt

# The slices will be ordered and plotted counter-clockwise.
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0) # only "explode" the 2nd slice (i.e. 'Hogs')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')

plt.show()


# ## 极坐标图--polar

# In[10]:

import numpy as np
import matplotlib.pyplot as plt

N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2 * np.random.rand(N)
colors = theta

ax = plt.subplot(111, polar=True)  # 画极坐标图，只需要设置一下子图中的polar=True即可
c = plt.scatter(theta, r, c=colors, s=area, edgecolor='k',cmap=plt.cm.hsv)
c.set_alpha(0.75)  # 点的透明度
plt.grid(False)  # 极坐标的网格不可见
# ax.set_rmax(max(r)+1)
plt.thetagrids([])  # 极坐标的角度刻度不可见
plt.axis('off')  # 关闭坐标轴
axcb = plt.colorbar()  # 生成colorbar

plt.show()

