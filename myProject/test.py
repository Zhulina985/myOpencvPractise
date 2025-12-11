#背景过滤
import operator

import cv2 as cv
import numpy as np
from fontTools.misc.cython import returns
from matplotlib.pyplot import imshow


#图像的处理
def select_color():

    #读取图片
    img=cv.imread("resource/img.png")

    # 二值化过滤背景
    #保留像素值为120-255
    lower=np.uint8([120,120,120])
    upper=np.uint8([255,255,255])

    #相当于增加掩膜
    white_mask=cv.inRange(img,lower,upper)
    masked_img = cv.bitwise_and(img, img, mask=white_mask)



    # Canny边缘检测

    low, high = 100, 180     #确定阈值
    img_can = cv.Canny(masked_img, low, high)

    return img_can



#这部分为选择所需要的区域，将非停车场的区域过滤
def select_region():

    #传入处理后的二值，进行过边缘检测的图片
    img=select_color()


    #手动选取主题部分
    p1=[40 ,240]
    p2=[40 ,325]
    p3=[580,325]
    p4=[580,35]
    p5=[360,35]
    p6=[220,170]

    vertice=np.array([[p1,p2,p3,p4,p5,p6]],dtype=np.int32)


    #选取点在实际的位置绘制(不必要)
    point_img=img.copy()
    point_img=cv.cvtColor(point_img,cv.COLOR_BGR2RGB)
    for point in vertice[0]:
        cv.circle(point_img,(int(point[0]),int(point[1])),2,(255,0,0),4)


    #mask，只保留内部区域
    mask=np.zeros_like(img)

    #图像过滤
    cv.fillPoly(mask,vertice,color=255)

    final_img=cv.bitwise_and(img,img,mask=mask)

    return final_img


#直线的初步选择
#最终以字典形式返回，特征为列数
def select_line():
    img=select_region()
    img_org=cv.imread("resource/img.png")

    #霍夫直线变换直接转变为直角坐标系
    lines=cv.HoughLinesP(img, rho=0.1, theta=np.pi/10, threshold=15, minLineLength=9, maxLineGap=4)
    prefect_line=[]

    #过滤直线
    for line in lines:
        x1,y1,x2,y2=line[0]


       # cv.line(img_org,(x1,y1),(x2,y2),(255,0,0),2)

        #将水平，长度符合的直线过滤出
        if abs(y2 - y1) <= 0.5 and 12 <= abs(x2 - x1) <= 35:
           cv.line(img_org,(x1,y1),(x2,y2),(0,0,255),1)
           prefect_line.append([x1,y1,x2,y2])


    #operator 先按x进行从小排序,再次基础上按照y进行从小排序，既可以做成第一列第一行，第二行、、、到第二列
    sort_lines=sorted(prefect_line,key=operator.itemgetter(0,1))


    #将排列好的直线用字典分类储存
    #按照列来，每列的车位进行整理
    #原理：每一列车列位置相差不大，相差太大为不同列
    diff_line={}
    list_row=[]
    label_row = 1
    for i in range(len(sort_lines)-1):
        a1,c1,a2,c2=sort_lines[i]
        b1,d1,b2,d2=sort_lines[i+1]


        if abs(a1-b1)<6:
            list_row.append(sort_lines[i])

        else:
            list_row.append(sort_lines[i])
            diff_line[label_row]=list_row
            label_row+=1
            list_row = []

    diff_line[label_row] = list_row
    label_row += 1
  #  cv.imshow('img',img_org)
  #  cv.waitKey(0)

    return diff_line


#将图片中的列型区域绘制成矩形
#
def rect_region():

    img=cv.imread("resource/img.png")
    dict_line=select_line()


    rect_lists=[]
    for key in dict_line.keys():
        values=dict_line[key]


        #讲一些非车位过滤(这里选定,车位大于5则为车位)
        if len(values)>5:
             values=sorted(values,key=lambda x:x[1])
             a=(values[0][0],values[0][1])
             b=(values[0][2]+10,values[-1][3])
             rect_lists.append([a,b])

             for rect_list in rect_lists:

                 cv.rectangle(img,rect_list[0],rect_list[1],(0,0,255),2)


    cv.imshow('img',img)
    cv.waitKey(0)

rect_region()



