import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread("resource/3c1.jpg",cv2.CV_16F) #读取图片，第二个为显示类型


def first():

    cv2.imshow("img",img)   #显示图片，第一个为窗口名
    cv2.waitKey(0)      #设定关闭时间，0即为关闭窗口，即图像一致停留

def colorRGB():

    img_b=img[ :, :,0]
    img_g=img[ :, :,1]
    img_r=img[ :, :,2]                  #BGR的色彩呈现是一个 M×N×3 的三维矩阵
    cv2.imshow("B",img_b)
    cv2.imshow("G",img_g)
    cv2.imshow("R",img_r)
    cv2.imshow("img",img)

    cv2.waitKey(0)

def reserve():
    img_plt=img[:,:,::-1]                   #反转BGR矩阵，得到RGB
    cv2.imshow("img_plt",img_plt)

    img_changH=img[:,::-1,:]
    cv2.imshow("img_chang",img_changH)   #水平翻转

    img_changV=img[::-1,:,:]
    cv2.imshow("img_chang",img_changV)    #竖直翻转
    cv2.waitKey(0)

def draw():
    # 创建一个全黑的三维矩阵作为图像对象
    img1 = np.zeros((512, 512, 3), np.uint8)
    # 绘制图形
    cv2.line(img1, (0, 0), (511, 511), color=(255, 0, 0), thickness=5)
    cv2.rectangle(img1, (384, 0), (510, 128), color=(0, 255, 0), thickness=3)
    cv2.circle(img1, (447, 63), radius=63, color=(0, 0, 255), thickness=-1)
    # 添加字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img1, "OpenCV", (10, 500), font, 4, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    # 显示图像
    plt.imshow(img1[:, :, ::-1])
    plt.title("result"), plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.waitKey(0)
draw()