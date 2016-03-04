# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

data_train=pd.read_csv("train.csv")
data_train.info()
data_train.describe()

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无")
plt.ylabel(u"人数")
plt.show()

e=data_train.Cabin.value_counts()
print e

fig=plt.figure()
# SibSp0=data_train.Survived[data_train.SibSp==0].value_counts()
# SibSp1=data_train.Survived[data_train.SibSp==1].value_counts()
# SibSp2=data_train.Survived[data_train.SibSp==2].value_counts()
# SibSp3=data_train.Survived[data_train.SibSp==3].value_counts()
# SibSp4=data_train.Survived[data_train.SibSp==4].value_counts()
# SibSp5=data_train.Survived[data_train.SibSp==5].value_counts()
# SibSp6=data_train.Survived[data_train.SibSp==6].value_counts()
# SibSp7=data_train.Survived[data_train.SibSp==7].value_counts()
# SibSp8=data_train.Survived[data_train.SibSp==8].value_counts()

p0=data_train.SibSp[data_train.Survived==0].value_counts()
p1=data_train.SibSp[data_train.Survived==1].value_counts()
df=pd.DataFrame({u'0':p0,u'1':p1})
# df=pd.DataFrame({u'0':SibSp0,u'1':SibSp1,u'2':SibSp2,u'3':SibSp3,u'4':SibSp4,u'5':SibSp5,u'6':SibSp6,u'7':SibSp7,u'8':SibSp8})
df.plot(kind='bar',stacked=True)
plt.show()

g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
h=data_train.groupby(['Parch','Survived'])
df1=pd.DataFrame(h.count()['PassengerId'])

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
data_train.Survived.value_counts().plot(kind='bar')# 柱状图
plt.title(u"获救情况 (1为获救)") # 标题
plt.ylabel(u"人数")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布 (1为获救)")

numPerson=len(data_train)
rmNullAge=[]
for i in range(numPerson):
    if data_train.Age[i]:
        rmNullAge.append(data_train.iloc[i])

print "rm Null Age",rmNullAge
a=pd.DataFrame(rmNullAge)
plt.subplot2grid((2,3),(1,0),colspan=2)
rmNullAge.plot
for i in range(numPerson):
    if data_train.Age[i]!=None:
        newData_train.append(data_train)
plt.subplot2grid((2,3),(1,0), colspan=2)
b1=data_train.Age[data_train.Pclass==1]
nb1=[]
for ele in b1:
    if ele!=np.NaN:
        nb1.append(ele)
b2=data_train.Age[data_train.Pclass==2]
b3=data_train.Age[data_train.Pclass==3]

nb1.plot(kind='kde')
b2.plot(kind='kde')
b3.plot(kind='kde')

#
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")# plots an axis lable
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")
plt.show()

#各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")
plt.show()


#按性别来查看获救情况
fig=plt.figure()
fig.set(alpha=0.2)

survived_male=data_train.Survived[data_train.Sex == 'male'].value_counts()
survived_female=data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'male':survived_male,u'female':survived_female})
df.plot(kind='bar',stacked=True)
plt.title(u"survived with sex")
plt.xlabel(u"sex")
plt.ylabel(u"number")
plt.show()


#各种船舱级别情况下获救情况
fig=plt.figure()
fig.set(alpha=0.7)

ax1=fig.add_subplot(121)
data_train.Survived[data_train.Pclass==3].value_counts().plot(kind='bar')
ax1.set_xticklabels([u"survived",u"unsurvived"],rotation=0)
ax1.set_yticklabels([u"number of person"])
plt.legend([u"high level seat"])

ax1=fig.add_subplot(122)
data_train.Survived[data_train.Pclass!=3].value_counts().plot(kind='bar',color='red')
ax1.set_xticklabels([u"survived",u"unsurvived"],rotation=0)
plt.legend([u"low level seat"])

plt.show()

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
print df
df.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口")
plt.ylabel(u"人数")

fig=plt.figure()
SibSp0=data_train.Survived[data_train.SibSp==0].value_counts()
SibSp1=data_train.Survived[data_train.SibSp==1].value_counts()
SibSp2=data_train.Survived[data_train.SibSp==2].value_counts()
SibSp3=data_train.Survived[data_train.SibSp==3].value_counts()
SibSp4=data_train.Survived[data_train.SibSp==4].value_counts()
SibSp5=data_train.Survived[data_train.SibSp==5].value_counts()
SibSp6=data_train.Survived[data_train.SibSp==6].value_counts()
SibSp7=data_train.Survived[data_train.SibSp==7].value_counts()
SibSp8=data_train.Survived[data_train.SibSp==8].value_counts()
df=pd.DataFrame({u'0':SibSp0,u'1':SibSp1,u'2':SibSp2,u'3':SibSp3,u'4':SibSp4,u'5':SibSp5,u'6':SibSp6,u'7':SibSp7,u'8':SibSp8})
df.plot(kind='bar',stacked=True)
plt.show()



