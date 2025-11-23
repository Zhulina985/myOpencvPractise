import sklearn
from sympy.physics.quantum.matrixutils import sparse


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


dir_catch()