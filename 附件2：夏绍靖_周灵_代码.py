# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:08:59 2020

@author: 周灵
"""
#网络购物购买意向预测的机器学习建模
#201630041 夏绍靖
#201630027 周灵

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score
import time

#导入数据
inputfile = 'H:/学习与学生工作资料/课件及作业/2019-2020-1/Python与大数据分析/期末大作业/online_shoppers_intention.csv'
df = pd.read_csv(inputfile)
#查看前五行数据和数据基本信息
print(df.head())
print(df.info())
#查看缺失值
df.isna().sum()
#对数据进行描述性统计
df.describe(include="all")
#对VisitorType、weekend数据做处理，转换为int数据
y=df['Revenue']
df['VisitorType'] = np.where(df['VisitorType'] == 'Returning_Visitor',1,0)
df['workday'] = np.where(df['Weekend']==True,0,1)
df.drop(['Weekend','Revenue','Month'], axis=1, inplace=True)
#查看指标的相关性
df.corr().round(2)
#去重
df.drop(['Administrative_Duration', 'Informational_Duration',
         'ProductRelated_Duration',
         'BounceRates'],axis=1, inplace=True)
#数据标准化处理
x = df
from sklearn import preprocessing
x = preprocessing.scale(x)
#划分训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


#SVM模型
from sklearn.svm import SVC
#获取学习器模型的参数列表和当前取值
SVC().get_params()

#使用核函数，设定gamma参数和惩罚系数取值范围
svc=SVC(kernel='rbf')
parameters={'gamma':[0.0001,0.0005,0.001],'C':[0.1,0.2,0.5,0.6,0.7,0.8,0.9,1,2,3,4]}
#使用网格搜索（GridSearchCV），确定精度最高的参数组合。
svc_grid=GridSearchCV(svc,param_grid=parameters,cv=5)

#训练SVM模型
svc_grid.fit(x,y)
#查看模型结果
svc_grid.cv_results_
#输出最优模型参数
print(svc_grid.best_params_)
print(svc_grid.best_score_)
#最优参数为：C=4，gamma=0.001

#用最优参数训练模型
svc=SVC(kernel='rbf',C=4,gamma= 0.001,probability=True)
svc.fit(x_train,y_train)
ypred=svc.predict(x_test)
yhat=svc.predict(x_train)
print("训练集精度:",np.mean(y_train==yhat))
print("测试集精度:",np.mean(y_test==ypred))
#模型评价
#文本分类报告
print(classification_report(y_test,ypred))
#weighted avg对应的精度（precision）为0.88，召回率（recall）为0.89，F1值为0.88
#AUC和ROC曲线
y_prob=svc.predict_proba(x_test)
fpr,tpr,_=roc_curve(y_test,y_prob[:,1])
plt.figure()
plt.plot(fpr,tpr,color='r',lw=2)
auc=roc_auc_score(y_test,y_prob[:,1])
print(auc)
#AUC值为0.8272

#决策树模型
#载入pydotplus包
%pip install pydotplus
#需要先在网络上下载Graphviz2.38包，解压后将其存放在“C:/Program Files (x86)”路径下
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz-2.38/bin/'

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
from IPython.core.display import Image

#获取学习器模型的参数列表和当前取值
DecisionTreeClassifier().get_params()

#由于标准化后数据集变为矩阵，将其转化为DataFrame
x1 = x.tolist()
x2 = pd.DataFrame(x1)
#给DataFrame命名，x2为标准化后的数据集（DataFrame数据结构）
x2.columns=["Administrative","Informational","ProductRelated","ExitRate","PageValues","SpecialDay","OperatingSystems","Browser","Region","TrafficTime","VisitorType","workday"]
x2

#先设定max_depth为3，看模型拟合结果
tree=DecisionTreeClassifier(max_depth=3)
tree.fit(x_train,y_train)
print("训练集精度:",tree.score(x_test,y_test))
print("测试集精度:",tree.score(x_train,y_train))
#max_depth为3时，tree.score为0.8913
#画出max_depth为3时的决策树模型
dot_data=StringIO()
export_graphviz(tree,
                out_file=dot_data,
                filled=True,
                feature_names=x2.columns,
                class_names=y.unique().astype(str),
                rounded=True,
                special_characters=True,
                )

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
#将决策树模型输出为pdf
graph.write_pdf("online shopping prediction1.pdf")

#寻找最优max_depth
tree=DecisionTreeClassifier()
parameters={'max_depth':np.arange(1,20,1)}
tree_grid=GridSearchCV(tree,param_grid=parameters,cv=5)
tree_grid.fit(x,y)
print(tree_grid.best_params_) 
print(tree_grid.best_score_)
#最优max_depth为5

#在最优max_depth下训练模型，评价模型效果
tree=DecisionTreeClassifier(max_depth=5)
tree.fit(x_train,y_train)
pre=tree.predict(x_test)
acc_train=tree.score(x_train,y_train)
acc_test=tree.score(x_test,y_test)
print("训练集精度:",acc_train)
print("测试集精度:",acc_test)
print(classification_report(y_test,pre))
#训练集精度为0.8991，测试集精度为0.8925

#画出最优max_depth对应的决策树模型
dot_data=StringIO()
export_graphviz(tree,
                out_file=dot_data,
                filled=True,
                feature_names=x2.columns,
                class_names=y.unique().astype(str),
                rounded=True,
                special_characters=True,
                )

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

graph.write_pdf("online shopping prediction2.pdf")

#查看决策树模型中各属性的重要性
stat=pd.DataFrame(columns=['importance','feature'])
stat['importance']=tree.feature_importances_
stat['feature']=x2.columns
stat.sort_values(by='importance',ascending=False,inplace=True)
print(stat.sort_values)
#pageValues重要性程度最高，为0.8986，其次是Administrative，ExitRate,ProductRelated,VisitorType等

#模型优化部分
#集成学习下的模型训练
tree=DecisionTreeClassifier(max_depth=2)
svc=SVC(kernel='rbf',C=4,gamma= 0.001,probability=True)

#基于Bagging构造集成学习（基学习器个数为 100）
from sklearn.ensemble import BaggingClassifier
bag_train_=[]
bag_test_=[]
bag_time=[]
for i in [svc,tree]:
    a=time.time()
    model=BaggingClassifier(base_estimator=i,n_estimators=100)
    model.fit(x_train,y_train) 
    bag_train = accuracy_score(y_train,model.predict(x_train))
    bag_test = accuracy_score(y_test,model.predict(x_test))
    print("Bagging train/test accuracies : %.3f/%.3f"%(bag_train,bag_test))
    bag_train_.append(str(bag_train)[:5])
    bag_test_.append(str(bag_test)[:5])
    bag_time.append(time.time()-a)
    
#基于AdaBoost构造集成学习（基学习器个数为 100）
from sklearn.ensemble import AdaBoostClassifier
ada_train_=[]
ada_test_=[]
ada_time=[]
for i in [svc,tree]:
    a=time.time()
    model=AdaBoostClassifier(base_estimator=i,n_estimators=100)
    model.fit(x_train,y_train) 
    ada_train = accuracy_score(y_train,model.predict(x_train))
    ada_test = accuracy_score(y_test,model.predict(x_test))
    print("AdaBoost train/test accuracies : %.3f/%.3f"%(ada_train,ada_test))
    ada_train_.append(str(ada_train)[:5])
    ada_test_.append(str(ada_test)[:5])
    ada_time.append(time.time()-a)
    
stat1={}
stat1['bag_train']=bag_train_
stat1['bag_test']=bag_test_
stat1['bag_time']=bag_time
stat1=pd.DataFrame(stat1,index=['svc','tree'])
print(stat1)

stat2={}
stat2['ada_train']=ada_train
stat2['ada_test']=ada_test_
stat2['ada_time']=ada_time
stat2=pd.DataFrame(stat2,index=['svc','tree'])
print(stat2)

#基于随机森林构造集成学习（基学习器个数为100）
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.get_params()

a=time.time()
model=RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train) 
rf_train = accuracy_score(y_train,model.predict(x_train))
rf_test = accuracy_score(y_test,model.predict(x_test))
rf_time=time.time()-a

stat3={}
stat3['rf_train']=rf_train
stat3['rf_test']=rf_test
stat3['rf_time']=rf_time
stat3=pd.DataFrame(stat3,index=['tree'])
print(stat3)


