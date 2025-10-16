import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


#mat模型


def moodle_2(img1,img2):
    fig,axes=plt.subplots(1,2)
    axes[0].imshow(img1[:,:,::-1])
    axes[1].imshow(img2[:,:,::-1])

    plt.show()

def moodle_4(img1,img2,img3,img4):
    fig,axes=plt.subplots(2,2)
    axes[0,0].imshow(img1[:,:,::-1])
    axes[0,1].imshow(img2[:,:,::-1])
    axes[1,0].imshow(img3[:,:,::-1])
    axes[1,1].imshow(img4[:,:,::-1])
    plt.show()

def start():
    img = cv.imread("resource/2024-09-16 215601.jpg", 0)

    plt.imshow(img, cmap='gray')  # 灰度图时只有一个通道，不是BGR型，只要img不要反转，后面是指定输出形式
    plt.show()

    cv.imwrite("download/test.jpg", img)  # 保存位置和名称，保存对象

def draw():
    img=np.zeros((512,512,3),np.uint8)  #512为大小，3是通道数量，后面那个灰度值

    cv.line(img, (0,0), (512,512), (255,0,0), 5)
    cv.circle(img,(200,200),50,(0,0,255),-1)        #最后是粗细，-1相当于实心圆
    cv.rectangle(img,(80,80),(180,180),(0,255,0),5)   #左上角和右下角

    cv.putText(img,"opencv",[400,400],4,cv.FONT_HERSHEY_PLAIN,(0,255,0),1)

    plt.imshow(img)       #BGR转RGB
    plt.show()

def change():

    img=np.zeros((512,512,3),np.uint8)
    cv.rectangle(img,(10,10),(500,500),(255,0,0),15)
    px=img[200,200]         #获取像素，直接用数列进行获取
    print(px)

    img[200,200]=[255,0,0]
    plt.imshow(img)
    px=img[200,200]
    plt.show()
    print(px)

    print(img.shape)
    print(img.dtype)
    print(img.size)  #512*512*3

    b,g,r=cv.split(img) #拆分通道
    img=cv.merge((b,g,r))   #合并通道

#操作
#分离，合并BRG，转变为灰度图
def ability():
    img=cv.imread("resource/and.jpg",1)

    b,r,g=cv.split(img)

    img=cv.merge((b,r,g))
    plt.imshow(img[:,:,::-1])

    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #转变灰色
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)   #转变为hsv
    plt.imshow(gray,cmap='gray')
    plt.show()

#d大小变化
def big():
    img=cv.imread("resource/and.jpg",1)
    img1= cv.resize(img,(512,512))  #绝对处理
    img2=cv.resize(img,None,fx=0.5,fy=0.5)  #相对处理

    r,l=img.shape[:2]   #获取两个值，得到长宽
    print(r)
    print(l)

    plt.imshow(img1[:,:,::-1])

#移动
def move():
    img=cv.imread("resource/and.jpg",1)
    M=np.float32([[1,0,100],[0,1,50]])      #传入2*3的矩阵向右下平移(100,50)
                                            #通用矩阵进行变换
    img1= cv.warpAffine(img,M,(img.shape[1],img.shape[0]))  #传入图片，矩阵，大小

    fig,axes=plt.subplots(1,2)
    axes[0].imshow(img[:,:,::-1])
    axes[0].set_title('original')
    axes[1].imshow(img1[:,:,::-1])
    axes[1].set_title('warped')
    plt.show()

#旋转
def row():
    img=cv.imread("resource/and.jpg",1)
    M=cv.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),90,1)  #旋转中心，角度放缩比例
    img1=cv.warpAffine(img,M,(img.shape[1],img.shape[0]))    #利用warp进行调整

    moodle_4(img,img1,img,img1)

#仿射
def map_change():

    img=cv.imread("resource/and.jpg",1)
    pts1=np.float32([[50,50],[200,50],[50,200]])        #需要两个矩阵进行转化
    pts2=np.float32([[100,100],[200,50],[100,250]])

    M=cv.getAffineTransform(pts1,pts2)                  #进行融合

    img1=cv.warpAffine(img,M,(img.shape[1],img.shape[0]))       #通过warpAffine进行图像的变化

    fig,axes=plt.subplots(1,2)
    axes[0].imshow(img[:,:,::-1])
    axes[0].set_title('original')
    axes[1].imshow(img1[:,:,::-1])
    axes[1].set_title('warped')

    plt.show()

#透射
def transmission():
    img=cv.imread("resource/and.jpg",1)
    pts1=np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2=np.float32([[100,145],[300,100],[80,290],[310,300]])   #4个矩阵
                                                                #同意perspective来改变

    M=cv.getPerspectiveTransform(pts1,pts2)

    img1=cv.warpPerspective(img,M,(img.shape[1],img.shape[0]))

    moodle_2(img,img1)

#总结:图像平移，旋转，放射，透射都通过矩阵进行，
# 其中移动为2个，旋转为中心和形状
#仿射为3*2，透射为4*2、

#图像金字塔——多分辨率解释图像,对原始图像进行不断采样
def gold():
    img=cv.imread("resource/and.jpg",1)
    img_up=cv.pyrUp(img)
    img_down=cv.pyrDown(img)

    cv.imshow('img',img)
    cv.imshow('img_up',img_up)   #上采样
    cv.imshow('img_down',img_down) #下采样
    cv.waitKey(0)

#形态学学习
#基于图像，二进制执行

#腐蚀和膨胀
def shap():
    img=cv.imread("resource/and.jpg",1)
    k=np.ones((3,3),np.uint8)   #创建核结构
    img2=cv.erode(img,k)        #腐蚀
    img3=cv.dilate(img,k)       #膨胀
    cv.imshow('img',img)
    cv.imshow('img2',img2)
    cv.imshow('img3',img3)
    cv.waitKey(0)


#开闭运算
def open_close():
    img=cv.imread("resource/and.jpg",1)
    k=np.ones((3,3),np.uint8)
    cvOpen=cv.morphologyEx(img,cv.MORPH_OPEN,k) #开操作，先腐蚀，去噪点高亮
    cvClose=cv.morphologyEx(img,cv.MORPH_CLOSE,k)   #闭操作，先膨胀，填充黑点
    cv.imshow('img',img)
    cv.imshow('cvOpen',cvOpen)
    cv.imshow('cvClose',cvClose)

    cv.waitKey(0)

#礼帽和黑帽运算
def hat():
    img=cv.imread("resource/and.jpg",1)
    k=np.ones((3,3),np.uint8)
    img2=cv.morphologyEx(img,cv.MORPH_TOPHAT,k) #分离较亮的点
    img3=cv.morphologyEx(img,cv.MORPH_BLACKHAT,k)   #分离较暗的点

    cv.imshow('img',img)
    cv.imshow('img2',img2)
    cv.imshow('img3',img3)

    cv.waitKey(0)

hat()