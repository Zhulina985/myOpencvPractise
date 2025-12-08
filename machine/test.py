import os
import tarfile
from heapq import merge
from importlib.resources import files
from os import write

import jieba
import sklearn
from scipy.stats import pearsonr
from sympy.physics.control.control_plots import matplotlib
from sympy.physics.quantum.matrixutils import sparse
import pandas as pd #读取文件

import matplotlib.pyplot as plt

from machine.fruit import x_train


#数据集获取
def start():
    sklearn.datasets.load_iris()    #鸢尾花数据集
    sklearn.datasets.load_iris()

  #  sklearn.datasets.fetch_iris()    #fetch 大数据集获取


    bunch = sklearn.datasets.load_iris()  #返回值是一个bunch类型，继承于字典



    print(bunch)
    print(bunch.feature_names)
    print(bunch.target_names)

#基本的数据集使用和模型分离
def iris_use():

    iris =  sklearn.datasets.load_iris()
    print("数据集描述：\n",iris["DESCR"])

    #训练集合测试集的划分
    #参数说明：特征值，目标值，测试集大小，随机种子(采样结果)
    #返回：特征值(用于训练，测试)，目标值(用于训练，测试)
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(iris.data, iris.target,test_size=0.2,random_state=22)

    print("x_train\n",x_train)
    print("x_test\n",x_test)
    print("y_train\n",y_train)
    print("y_test\n",y_test)


#字典的转化
def dir_catch():

    #向量——矩阵——二维数组
    #向量化，即为转化为数组形式
    feature1=[{'city':"北京","tem":12},
              {"city":"上海","tem":14},
              {"city":"广州","tem":17},
              {"city": "苏州", "tem": 16}
        ,
              {"city": "杭州", "tem": 16}
        ,
              {"city": "黑龙江", "tem": 6}


              ]

    #特征值转化
    #默认返回一个sparse(稀疏矩阵)
    vet=sklearn.feature_extraction.DictVectorizer().fit_transform(feature1)
    #修改后
    vet1=sklearn.feature_extraction.DictVectorizer(sparse=False).fit_transform(feature1)
    print(vet1)

    feat=sklearn.feature_extraction.DictVectorizer(sparse=False)



#字符串文本的特征提取
def test_catch():

    #特征，每一个单词记一个数字。相同的单词累加,一个字母不算
    #英文
    s=["life is is a flower"]
    tran=sklearn.feature_extraction.text.CountVectorizer()

    data1=tran.fit_transform(s)

    print(data1.toarray())       #sparse矩阵转化为一般二维数组，统计特征词出现个数

    #中文
    #需要进行分词处理，单个字符无法转化
    c=["生活 是 一朵花 一朵花 生活"]

    data2=tran.fit_transform(c)
    print("data2",data2)


    data3=tran.fit_transform(s)

    print(tran.get_feature_names_out()) #根据最后一次fit_transform进行转化


    #停用词的使用
    #is不会被记录
    tran1=sklearn.feature_extraction.text.CountVectorizer(stop_words=['is'])
    data4=tran1.fit_transform(s)
    print(data4.toarray())
    print(tran1.get_feature_names_out())


#中文文本的分词实现
def count_chinese():

    #运用jieba库进行分词处理
    p=["数据挖掘是指从大量数据中发现有价值的隐藏模式、发现规律和发现知识的过程"]

    cut=list(jieba.cut(p[0]))   #需要将jieba对象进行强转,得到有字符串组成的数列
    print(cut)

    text=" ".join(cut)          #进行合并操作,得到空格分开的字符串
    print(text)
    tran=sklearn.feature_extraction.text.CountVectorizer()
    data1=tran.fit_transform([text])
    print(data1.toarray())
    print(tran.get_feature_names_out())


#用tf_idf进行处理  评估关键词
def ti_idf():

    p = ["数据挖掘是指从大量数据中发现有价值的隐藏模式、规律知识，知识和知识的过程"]

    cut = list(jieba.cut(p[0]))
    print(cut)

    text = " ".join(cut)
    print(text)
    tran = sklearn.feature_extraction.text.TfidfVectorizer()        #得到相当于权重信息
    data1 = tran.fit_transform([text])
    print(data1.toarray())
    print(tran.get_feature_names_out())


#无量纲化处理
#数据的归一化以及标准化
def minmax():
    #获取数据
    #实例化转换器

    data=pd.read_csv("text.txt")

    data_get=data.iloc[:,:3]
       #iloc对文本数据进行相数组一样的处理

    trans=sklearn.preprocessing.MinMaxScaler()      #归一化转换器，以最值来处理
    trans1=sklearn.preprocessing.StandardScaler()   #标准化转换器，以正态分布处理(不易受到异常处理，更稳定，鲁棒性强)

    data_new=trans.fit_transform(data_get)
    data_new1=trans1.fit_transform(data_get)
    print("minmax",data_new)
    print("stand",data_new1)


#降维学习
#低方差过滤
def variance():
    data=pd.read_csv("data1.txt")
    data_get=data.iloc[:,1:-2]


    tran=sklearn.feature_selection.VarianceThreshold(threshold=5)
    data_new=tran.fit_transform(data_get)
    print(data_new)

#求线性相关系数
def relation():
    data=pd.read_csv("data1.txt")
    data_get=data.iloc[:,1:3]
    data1=data.iloc[:,1:2]
    data2=data.iloc[:,2:3]

    r=pearsonr(data["pe_ratio"],data["pb_ratio"])   #传入连续数据进行相关系数的判断，越小越不相关


    print(r)

    plt.scatter(data["pe_ratio"],data["pb_ratio"],color="red")      #散点图绘制，发现不是很线性
    plt.xlabel("pe_ratio")
    plt.ylabel("pb_ratio")
    plt.show()

#主成分分析
#降维的同时保留更多的原信息
def PCA_demo():
    data=[[1,2,3],[5,3,2],[8,3,5]]
    tran=sklearn.decomposition.PCA(n_components=0.98)    #参数为整数，为保护的个数，小数时，保留的百分比
    data_new=tran.fit_transform(data)

    print(data_new)


#案例分析
#market basket analyse
#目标:合并表格数据，对数据进行降维处理
#总结步骤:读取表格,合并表格,生成交叉表格,PCA处理数据
def case():

    #找到use_id和aisle_id的关系
    #交叉表的绘制
    data=pd.read_csv("resource/table_final.csv")
    table=pd.crosstab(data["user_id"],data["aisle"])        #这里得到(206209,134)的数组

    #降维处理

    tran=sklearn.decomposition.PCA(n_components=0.95)      #PCA处理后得到(206209,44)的数组,去除了很多无关部分
    new_data=tran.fit_transform(table)

    print(new_data)
    print(new_data.shape)


#读取,合并表阶段,直接保存到本地,后续省去读取部分
def front_read():
    # 读取数据

    data_other_product = pd.read_csv("resource/order_products__prior.csv")
    data_product = pd.read_csv("resource/products.csv")
    data_others = pd.read_csv("resource/orders.csv")
    data_aisles = pd.read_csv("resource/aisles.csv")

    # 合并数据
    # 将product_id (在other_product和product) 与 aisles_id(在aisles,product)进行合并

    tabel_aisles = pd.merge(data_aisles, data_product, on=["aisle_id", "aisle_id"])

    table_product_aisles = pd.merge(tabel_aisles, data_other_product, on=["product_id", "product_id"])

    table_final = pd.merge(table_product_aisles, data_others, on=["order_id", "order_id"])
    # 合并得到最终的表格

    #保存到本地
    table_final.to_csv("resource/table_final.csv", index=False)




#算法分类
#KNN算法实例
#鸢尾花数据集分类
def knn_iris():

    #1.获取数据

    iris=sklearn.datasets.load_iris()

    #2.划分数据集
    #特征值和目标值
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(iris.data,iris.target)

    #3.标准化处理
    tran=sklearn.preprocessing.StandardScaler()
    x_train_st=tran.fit_transform(x_train)
    x_test_st=tran.transform(x_test)      #这里直接用transform,是保证对训练集和测试集进行相同的操作

    #4.KNN算法训练模型
    estimator=sklearn.neighbors.KNeighborsClassifier()

    # 加入网格搜索(参数:预估器，n值的可能值,折叠次数)
    estimator=sklearn.model_selection.GridSearchCV(estimator,param_grid={"n_neighbors":[1,2,3,4,5,6,7,8,9,10]},cv=10)


    estimator.fit(x_train_st,y_train)       #得到模型estimator


    #5.模型评估
    #通过传入测试集数据来获得预测数据集
    predicted=estimator.predict(x_test_st)
    print("直接比对",predicted)
    print("是否相等",y_test==predicted)
    score=estimator.score(x_test_st,y_test)     #准确率计算,相当于传入真实值(测试与目标)与模型本身进行比较,计算得出的结果 (整体的结果)
    print("准确率",score)

    print("最佳参数",estimator.best_params_)      #训练集内部的结果
    print("最佳结果",estimator.best_score_)




#facebook签到案例
#签到位置的预测
def case_facebook():

    #读取数据
    data=pd.read_csv("resource/facebook.csv")

    #对数据的特征值和目标值进行确定

    x=data[["x","y","accuracy","time"]]
    y=data["place_id"]

    #进行划分
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y)


    #标准化处理
    tran=sklearn.preprocessing.StandardScaler()
    x_train_st=tran.fit_transform(x_train)
    x_test_st=tran.transform(x_test)

    #生成预测器
    es=sklearn.neighbors.KNeighborsClassifier()
    es=sklearn.model_selection.GridSearchCV(es,param_grid={"n_neighbors":[1,2,3,4,5,6,7,8,9,10]},cv=10)

    es.fit(x_train_st,y_train)
    predicted=es.predict(x_test_st)

    print(es.best_params_)

    print(predicted==y_test)

    print(es.score(x_test_st,y_test))


#朴素贝叶斯的学习
#对新闻进行分类
#运用sklearn自带数据集
def bayes_iris():

    print(1)



    #1.获取数据(数据集较大，运用fetch)
    data_news=sklearn.datasets.fetch_20newsgroups()

    print(data_news)

    #2.划分数据集

    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(data_news.data,data_news.target)

    #3.进行特征抽取,返回sparse矩阵

    transfer=sklearn.feature_extraction.text.CountVectorizer()

    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    #4.引入朴素贝叶斯算法

    estimator=sklearn.naive_bayes.MultinomialNB(alpha=1.0)  #拉普拉斯平滑系数

    estimator.fit(x_train,y_train)
    predicted=estimator.predict(x_test)
    print(predicted==y_test)
    print(predicted)

    mark=estimator.score(x_test,y_test)
    print(mark)

bayes_iris()






