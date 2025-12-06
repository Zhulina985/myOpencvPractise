import os
import cv2 as cv
import numpy as np
import sklearn

X=[]
Y=[]


for i in range(2):
    if i==0:
        label1="Cat"
    else:
        label1="Dog"

    for j in range(500):

        img_or=cv.imread("animals/"+label1+"/"+str(j)+".jpg",1)

        img_resize=cv.resize(img_or,(250,250),interpolation=cv.INTER_CUBIC)
        hist=cv.calcHist([img_resize],[0],None,[256],[0,256])

        X.append((hist/255).flatten())
        Y.append(label1)

X=np.array(X)
Y=np.array(Y)

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.2)
tran=sklearn.preprocessing.StandardScaler()
x_train_st=tran.fit_transform(x_train)
x_test_st=tran.transform(x_test)


estimate=sklearn.neighbors.KNeighborsClassifier(n_neighbors=20)
estimate.fit(x_train_st,y_train)



predicted=estimate.predict(x_test_st)
print(predicted==y_test)
print(estimate.score(x_test_st,y_test))