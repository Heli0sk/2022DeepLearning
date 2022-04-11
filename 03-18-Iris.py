import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

columns = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Species"]
dataset = pd.read_table('data/iris.data',sep=',',header=None, names=columns)
print(dataset.describe)
# 显示了每一对特征之间的双变量关系
sns.pairplot(dataset, hue="Species", size=3)
plt.show()

# 使用sklearn进行分类
data = dataset.values
X = data[:, 0:3] #特征
Y = data[:, 4] #类别
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=66)
#使用默认参数，如果数据比较复杂的话需要调参
clas = GradientBoostingClassifier(random_state=58)
clas.fit(X_train,Y_train)
Y_pre1=clas.predict(X_train)
Y_pre2=clas.predict(X_test)
print("训练集准确率：%s"%clas.score(X_train,Y_train))
print("测试集准确率：%s"%clas.score(X_test,Y_test))
print("report is:",classification_report(Y_test,Y_pre2))