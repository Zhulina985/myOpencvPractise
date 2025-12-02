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
    #参数说明：特征值，标签值，测试集大小，随机种子(采样结果)
    #返回：特征值(训练，测试)，目标值(训练，测试)
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
    table_final.to_csv("resource/table_final.csv", index=False)


case()
