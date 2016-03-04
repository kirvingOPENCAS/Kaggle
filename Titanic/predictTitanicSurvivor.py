# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from regression import *
from sklearn import linear_model
import sklearn.preprocessing as preprocessing

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # y即目标年龄
    y = known_age[:, 0]
    # X即特征属性值
    X = known_age[:, 1:]
    testX=unknown_age[:,1:]
    # predictedAges=lwlrTest(testX,X,y,k=0.4)
    # print predictedAges
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()),'Age']=predictedAges

    return df,rfr

#将Cabin简单分为 yes 与 no 两个类别
def set_Cabin_type(df):

    df.loc[(df.Cabin.isnull()),'Cabin']='yes'
    df.loc[(df.Cabin.notnull()),'Cabin']='no'

    return df

def dataPreprocessing(filename):

    data_train=pd.read_csv(filename)
    data_train,rfr = set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)

#将类别数据离散化
    dummies_Cabin=pd.get_dummies(data_train['Cabin'],prefix='Cabin')
    dummies_Embarked=pd.get_dummies(data_train['Embarked'],prefix='Embarked')
    dummies_Pclass=pd.get_dummies(data_train['Pclass'],prefix='Pclass')
    dummies_Sex=pd.get_dummies(data_train['Sex'],prefix='Sex')

#将原始的 Cabin Embarked Pclass Sex删除
    data_train.drop(['Cabin','Embarked','Pclass','Sex'],axis=1,inplace=True)

#构造新的dataFrame
    df=pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis=1)

#将 Age 与 Fare做归一化处理,利用sklearn 中的preprocessing模块
#实例化一个StandardScaler对象
    ps=preprocessing.StandardScaler()
    Age_scale_param=ps.fit(df['Age'])
    df['Age_scaled']=ps.fit_transform(df['Age'],Age_scale_param)
    Fare_scale_param=ps.fit(df['Fare'])
    df['Fare_scaled']=ps.fit_transform(df['Fare'],Fare_scale_param)

    return df,rfr

def trainLR(trainDF):
#数据预处理完成，用logistic来对数据进行训练
    train_df=trainDF.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np=train_df.as_matrix()
    trainAttr=train_np[:,1:]
    trainLab=train_np[:,0]
    lrModel=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
    lrModel.fit(trainAttr,trainLab)
    return lrModel

def testLR(filename,lrModel,RFModel):
    #用测试数据进行测试
    data_test=pd.read_csv(filename)
    data_test.loc[(data_test.Fare.isnull()),'Fare']=0
    age_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    unknowAge=age_df[data_test.Age.isnull()].as_matrix()

    #对测试数据进行与训练数据同样的处理
    #首先是补全年龄
    predictedAge=RFModel.predict(unknowAge[:,1::])
    data_test.loc[(data_test.Age.isnull()),'Age']=predictedAge

    #将Cabin值量化为 yes 与 no
    data_test=set_Cabin_type(data_test)
    #将Sex,Pclass,Embarked,Cabin等量化为数值
    dummies_Cabin=pd.get_dummies(data_test['Cabin'],prefix='Cabin')
    dummies_Pclass=pd.get_dummies(data_test['Pclass'],prefix='Pclass')
    dummies_Embarked=pd.get_dummies(data_test['Embarked'],prefix='Embarked')
    dummies_Sex=pd.get_dummies(data_test['Sex'],prefix='Sex')
    #将原始Cabin,Pclass,Embarked,Sex删除
    data_test.drop(['Cabin','Pclass','Embarked','Sex'],axis=1,inplace=True)
    #拼接新的数据
    df_test=pd.concat([data_test,dummies_Cabin,dummies_Pclass,dummies_Embarked,dummies_Sex],axis=1)
    #将Age、Fare做归一化处理
    ps=preprocessing.StandardScaler()
    Age_scale_param=ps.fit(df_test['Age'])
    df_test['Age_scaled']=ps.fit_transform(df_test['Age'],Age_scale_param)
    Fare_scale_param=ps.fit(df_test['Fare'])
    df_test['Fare_scaled']=ps.fit_transform(df_test['Fare'],Fare_scale_param)

    #利用训练好的模型进行预测
    #利用正则表达式提取需要的属性值
    test_df=df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictRes=lrModel.predict(test_df)
    result=pd.DataFrame({'PassengerID':df_test['PassengerId'].as_matrix(),'Survived':predictRes.astype(np.int32)})
    result.to_csv('/home/hadoop/PycharmProjects/Titanic/predictSuivivedResult.csv',index=False)
    return

trainDataFrame,randomForestModel=dataPreprocessing('train.csv')
LogisticRegressionModel=trainLR(trainDataFrame)
testLR('test.csv',LogisticRegressionModel,randomForestModel)
