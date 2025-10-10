import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

def paper():
    ax=fig.add_subplot(111)     #对于上面的fig.add_subplot(111)就是添加Axes的，参数的解释的在画板的第1行第1列的第一个位置生成一个Axes对象来准备作画
    ax.set(xlim=(0,1), ylim=(0,1),title='title',    #长度，表格名
    ylabel="Y",xlabel="X")  #x,y轴的名称

    plt.show()

def other():

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    plt.show()

    #归纳：fig为版面，ax为AXES即为画板
    #前面两个参数确定了面板的划分，例如 2， 2会将整个面板划分成 2 * 2 的方格，第三个参数取值范围是 [1, 2*2]，即用第三个参数确定坐标

def adv():
    fig1, axes = plt.subplots(nrows=2, ncols=2)#创建二维数组
    axes[0, 0].set(title='Upper Left')
    axes[0, 1].set(title='Upper Right')
    axes[1, 0].set(title='Lower Left')
    axes[1, 1].set(title='Lower Right')
    plt.show()
    #利于循环快速构图
def pyplot():
    plt.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
    plt.xlim(0.5, 4.5)
    plt.show()

def line():
    x=np.linspace(0,np.pi) #x的范围
    y=np.sin(x)
    z=np.cos(x)
    ax1= plt.plot(x, y,z)
    plt.show()

def keyword():
    x=np.linspace(0,10,200)
    data_object={'x': x,
            'y1': 2 * x + 1,
            'y2': 3 * x + 1.2,
            'mean': 0.5 * x * np.cos(2*x) + 2.5 * x + 1.1}  #该字典为一个数据集，可以被axes解读
    fig1, axes, = plt.subplots(nrows=2, ncols=2)


    axes[0,0].plot(x,"mean",color="blue",data=data_object)    #横轴，纵轴，颜色，传递数据集，其中前两者均从数据集中得到
    plt.show()
keyword()