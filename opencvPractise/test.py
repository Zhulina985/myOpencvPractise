import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sympy import false


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

#图像平滑
#噪声处理：椒盐和高斯噪声
#椒盐：黑白点
#高斯：噪声满足高斯公式(正态分布)

#均值滤波:算法较快，但细节保留度差，马赛克
#高斯滤波：通过创建权重矩阵，通过正态分布分配权重，计算得到高斯模糊的值
#中值滤波：同意邻域内像素点中值替代原灰度值，主要作用于椒盐噪声（极端的黑白点被中值替代）
def average_gao_mid_noise():
    img=cv.imread("resource/and.jpg",1)
    blur1=cv.blur(img,(10,10))    #创建卷积核
    blur2=cv.GaussianBlur(img,(9,9),1)  #权重矩阵需要中心点，所以必须为奇数
    blur3=cv.medianBlur(img,3)  #因为是中值，所以卷积核就一个数
    cv.imshow('blur1',blur1)
    cv.imshow('blur2',blur2)
    cv.imshow('blur3',blur3)
    cv.imshow('img',img)
    cv.waitKey(0)


#直方图和掩膜学习
#直方图
def histogram():
    img=cv.imread("resource/and.jpg",0) #以计算灰度直方图为例，传入灰度图像
    hist=cv.calcHist([img],[0],None,[256],[0,256])  #参数要中括号，0为灰度直方图，掩膜，BIN数目，范围大小

    plt.plot(hist)

    plt.grid()  #图像网格添加
    plt.show()


#掩膜:直方图中的mask参数
#对遮挡区域不处理。遮挡过程为掩膜
#用二维矩阵进行掩膜
#相当于选取区域
def mask():
    img=cv.imread("resource/and.jpg",0)
    mask1=np.zeros(img.shape[:2],np.uint8)  #确定蒙版
    mask1[400:650,200:500]=1                #设置区域参数，高400-650 宽200-500

    mask_img=cv.bitwise_and(img,img,mask=mask1) #传入图片和蒙版(进行与操作)
    mask_histr=cv.calcHist([img],[0],mask1,[256],[0,256])   #绘制直方图，掩膜存在mask=mask1

    fig,axes=plt.subplots(2,2)
    axes[0,0].imshow(img,cmap='gray')
    axes[0,0].set_title('original')
    axes[0,1].imshow(mask1,cmap='gray')
    axes[0,1].set_title('mask')
    axes[1,0].imshow(mask_img,cmap='gray')
    axes[1,0].set_title('mask image')
    axes[1,1].plot(mask_histr)
    axes[1,1].grid(True)
    axes[1,1].set_title('histogram')
    plt.show()

#直方图均衡化
#将直方图进行横向拉升，扩大像素的分发区域，提高对比度

def average_histogram():
    img=cv.imread("resource/and.jpg",0)
    hist=cv.equalizeHist(img)   #均衡化后得到图像

    hist1=cv.calcHist([img],[0],None,[256],[0,256])
    hist2=cv.calcHist([hist],[0],None,[256],[0,256])

    fig,axes=plt.subplots(2,2)
    axes[0,0].imshow(img,cmap='gray')
    axes[0,0].set_title('original')
    axes[0,1].imshow(hist,cmap='gray')
    axes[0,1].set_title('histogram')
    axes[1,0].plot(hist1)
    axes[1,0].grid(True)
    axes[1,0].set_title('average histogram')
    axes[1,1].plot(hist2)
    axes[1,1].grid(True)
    axes[1,1].set_title('average histogram')

    plt.show()

#自适应直方图均衡化
#均衡化后，直方图可能会过曝或过暗，所以要进行分块处理
#设定灰度限制并用起进行补充
def auto_average_histogram():
    img=cv.imread("resource/and.jpg",0)
    clahe=cv.createCLAHE(clipLimit=1, tileGridSize=(8,8))   #输入限制和分块参数，创建一个自动化对象
    img1=clahe.apply(img)       #进行应用

    hist1=cv.calcHist([img],[0],None,[256],[0,256])
    hist2=cv.calcHist([img1],[0],None,[256],[0,256])

    fig,axes=plt.subplots(2,2)
    axes[0,0].imshow(img,cmap='gray')
    axes[0,0].set_title('original')
    axes[0,1].imshow(img1,cmap='gray')
    axes[0,1].set_title('histogram')
    axes[1,0].plot(hist1)
    axes[1,0].grid(True)
    axes[1,0].set_title('average histogram')
    axes[1,1].plot(hist2)
    axes[1,1].grid(True)
    axes[1,1].set_title('average histogram')
    plt.show()


#边缘检测
#利用各算子进行检测轮廓，其中核心为导数
#sobel算子，求一阶导数
def sobel():
    img=cv.imread("resource/and.jpg",0)
    x=cv.Sobel(img,cv.CV_64F,1,0)       #利用算子，对x,y方向各进行检测，得到两张(类似高位反差保留)
    y=cv.Sobel(img,cv.CV_64F,0,1)       #图像，图像深度，x,y方向

    x1 = cv.Sobel(img, cv.CV_64F, 1, 0,ksize=-1)    #ksize设为-1，利用的是scharr算子
    y1 = cv.Sobel(img, cv.CV_64F, 0, 1,ksize=-1)

    sobel_x=cv.convertScaleAbs(x)               #格式转化,得到8位图
    sobel_y=cv.convertScaleAbs(y)

    sobel_x1=cv.convertScaleAbs(x1)
    sobel_y1=cv.convertScaleAbs(y1)

    result=cv.addWeighted(sobel_x,0.5,sobel_y,0.5,0)            #利用权重函数，进行图像的融合，x,y方向各取0.5,得到边缘检测结果
    result1=cv.addWeighted(sobel_x1,0.5,sobel_y1,0.5,0)



    fig,axes=plt.subplots(1,3)
    axes[0].imshow(sobel_x,cmap='gray')
    axes[0].set_title('original')
    axes[1].imshow(sobel_y,cmap='gray')
    axes[1].set_title('result')
    axes[2].imshow(result1,cmap='gray')
    axes[2].set_title('result1')

    plt.show()

#拉普拉斯算子
#对导数进行二阶求导
#相较于sobel算子，边界更加深度，像素更少
def laplacian():
    img=cv.imread("resource/and.jpg",0)
    result=cv.Laplacian(img,cv.CV_64F)  #利用拉普拉斯函数进行检测
    la_abs=cv.convertScaleAbs(result)   #格式转换，转成8位类型

    fig,axes=plt.subplots(1,2)
    axes[0].imshow(img,cmap='gray')
    axes[0].set_title('original')
    axes[1].imshow(la_abs,cmap='gray')
    axes[1].set_title('result')
    plt.show()

#canny边缘检测
#四步法，核心通过最大梯度来判断是否为边界
def canny():
    img=cv.imread("resource/and.jpg",0)
    result=cv.Canny(img,100,100)    #canny函数,阈值1：连接间断边缘  阈值2：检测明显边缘

    fig,axes=plt.subplots(1,2)
    axes[0].imshow(img,cmap='gray')
    axes[0].set_title('original')
    axes[1].imshow(result,cmap='gray')
    axes[1].set_title('result')
    plt.show()


#模板匹配
#通过一张图片，在另一张图片上面寻找匹配的区域
def match():
    img=cv.imread("resource/and.jpg")       #需要匹配的图片
    template=cv.imread("resource/and1.jpg") #匹配模板
    h,w,l=template.shape                    #得到模板的长宽和深度

    res=cv.matchTemplate(img,template,cv.TM_CCOEFF)     #通过match方法进行匹配，最后参数为匹配实现算法
    a,b,c,d=cv.minMaxLoc(res)                           #得到结果区域

    print(a,b,c,d)

    top_left=d      #d为左上角图像，已它为基准
    bottom_right=(top_left[0]+w,top_left[1]+h)  #加上模板的长宽得到右下角左边
    print(top_left,bottom_right)
    cv.rectangle(img,top_left,bottom_right,(0,255,0),3) #绘制图像
    cv.rectangle(img,d,c,(0,0,255),2)
    plt.imshow(img[:,:,::-1])
    plt.title("res"),plt.xticks([]),plt.yticks([]) #不显示x,y抽数据

    plt.show()


#霍夫变换


#线检测
#先变为2值图像，或先进行边缘检测
def hungarian():
    img=cv.imread("resource/huo.png")
    res=cv.Canny(img,50,100)        #进行Canny边缘检测

    lines=cv.HoughLines(res,1,np.pi/180,150)        #霍夫线检测
                                                                  #需要检测边缘，传入极坐标参数，以及检测参数
    for line in lines:              #极坐标的参数和图像
        rho,theta=line[0]           #遍历所有直线
        a=np.cos(theta)             #通过cos和sin进行转化
        b=np.sin(theta)
        x0=a*rho
        y0=b*rho
        x1=int(x0+1000*(-b))            #转变参数，转变为笛卡尔坐标系
        y1=int(y0+1000*(a))
        x2=int(x0-1000*(-b))
        y2=int(y0-1000*(a))             #
        cv.line(img,(x1,y1),(x2,y2),(255,0,0),2)

    plt.figure(figsize=(10,8),dpi=80)
    plt.imshow(img,cmap='gray')
    plt.show()


#圆检测
#内部进行滤波，再进行检测
def hug_circle():
    img=cv.imread("resource/huo.png")
    img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)                 #转为灰度图
    img1=cv.medianBlur(img_gray,7)                         #经行中值滤波，减少影响

    #进行霍夫检测
    #滤波处理后的图片，分辨率，圆心距离(数值内认为同一圆心)，边缘检测的参数，圆心和半径的共有阈值，最小半径，最大半径
    circle=cv.HoughCircles(img1,cv.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius=10,maxRadius=1000)

    #得到一个三维数组
    print(circle)

    #这里循环这样理解：
    #取三维组第一个值，在这个得到的二维数组中进行遍历。每一组即为一个圆的信息
    for i in circle[0,:]:

        print(i[0],i[1],i[2])


        #绘制圆
        #圆心为0,1，半径为2    注意：由于opencv的更新，需要强制转换成int数值，才能正常运行
        cv.circle(img,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),3)

        #绘制圆心，半径取足够小
        cv.circle(img,(int(i[0]),int(i[1])),2,(0,0,255),3)

    plt.figure(figsize=(10, 8), dpi=80)
    plt.imshow(img[:,:,::-1], cmap='gray')
    plt.title("圆检测")

    plt.show()


#角点检测
#harris检测,通过椭圆(缓慢变化和快速变化).得到灰度变化最快的部分
def corner1():
    img=cv.imread("resource/corner.png")


    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    gray=np.float32(gray)   #将图片转为float32类型

    dst=cv.cornerHarris(gray,3,3,0.04)  #进行harriers检测,邻域,sobel核大小,检测参数(0.04-0.05)

       #得到角点信息

    img[dst>0.002*dst.max()]=[0,0,255]      #设置阈值,打印出所有角点  ?运行原理?

    plt.figure(figsize=(10, 8), dpi=80)
    plt.imshow(img[:,:,::-1])
    plt.title("角点检测")
    plt.show()

#tomas检查
#harris检测升级,用M矩阵的特征值，已两侧检测中较小值为基准，其大于阈值，则说明是角点
def corner2():
    img=cv.imread("resource/corner.png")
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    corner=cv.goodFeaturesToTrack(gray,100,0.0000001,0.05,mask=None)        #tomas检测api,图像,最大角点数，角点着质量(0-1),最小间隔距离

    print(corner[0].ravel())       #得到坐标

    for i in corner:
        x,y=i.ravel()
        x=int(x)
        y=int(y)
        cv.circle(img,(x,y),2,(0,0,255),-1)     #循环绘制图像

    plt.figure(figsize=(10, 8), dpi=80)
    plt.imshow(img[:,:,::-1])
    plt.show()


#上面两种算法具有旋转不变性：旋转图像依然能检测出来
#不具备尺度不变型：由于上述两种算法的本质为利用很小区域，检测该区域中的灰度差值变化，所以图像的大小会影响检测结果
#sift关键点的检测
def sift():


    img=cv.imread("resource/corner.png")

    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    sift1=cv.SIFT.create()       #创建sift算法对象

    kp,des=sift1.detectAndCompute(image=gray,mask=None)     #kp:关键点(位置，尺度，方向信息)  des:描述，梯度信息

    # 绘制关键点:原始图像，关键点，输出图像,颜色(默认多色),flag:绘图的表示功能(绘制样式)
    cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(figsize=(10, 8), dpi=80)
    plt.imshow(img[:,:,::-1])
    plt.show()



#fast检测
#对算法进行优化，在选取点中的圆型区域检测阈值，优点：速度快 缺点：关键点过多
def fast():
    img=cv.imread("resource/corner.png")

    #开启非极大值抑制
    fast=cv.FastFeatureDetector()

    kp=fast.detect(img,None)

    img1=cv.drawKeypoints(img,kp,img,(0,0,255))

    #关闭抑制
    fast.setNonmaxSuppression(false)
    kp=fast.detect(img,None)

    img2=cv.drawKeypoints(img,kp,img,(0,0,255))

    fig,axes=plt.subplots(nrows=1,ncols=2)
    axes[0].imshow(img1[:,:,::-1])
    axes[1].imshow(img2[:,:,::-1])

    plt.show()

#orb检测

def orb():

    img=cv.imread("resource/corner.png")

    orb=cv.ORB()

    kp,des=orb.detectAndCompute(img,None)

    img2=cv.drawKeypoints(img,kp,img,(0,0,255))

    plt.figure(figsize=(10, 8), dpi=80)
    plt.imshow(img2[:,:,::-1])
    plt.show()



#视频流操作
def video():

    video=cv.VideoCapture("resource/videomp4.mp4")  #视频读取

    while(video.isOpened()):                        #检测是否成功
        ret,frame=video.read()                      #读取每一帧
        if ret==True:                               #检测是否成功
            cv.imshow('frame',frame)        #展示
        if cv.waitKey(25) & 0xFF == ord('q'):        #设置帧率并设置退出按键
            break


    video.release()             #释放视频
    cv.destroyAllWindows()

def video_set():
   video=cv.VideoCapture("resource/videomp4.mp4")
   width = int(video.get(3))
   height = int(video.get(4))

   print(width,height)
   fourcc = cv.VideoWriter_fourcc(*'mp4v')      #设置视频4字节码
   out=cv.VideoWriter("download/test.mp4",fourcc,60.0,(width,height))       #保存位置、字节码、帧率、长宽

   while True:
       ret,frame=video.read()

       if ret:
           out.write(frame)             #保存
           cv.imshow('frame',frame)
           if cv.waitKey(25) & 0xFF == ord('q'): break


   video.release()
   out.release()
   cv.destroyAllWindows()


video_set()
