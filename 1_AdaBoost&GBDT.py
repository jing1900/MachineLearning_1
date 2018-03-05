from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
import numpy as np

#训练集
iris = datasets.load_iris()
x = iris.data[:,:2]
y = iris.target

#构建模型
m1 = DecisionTreeClassifier(max_depth=5)
m2 = AdaBoostClassifier(m1,n_estimators = 100)
m3 = GradientBoostingClassifier(n_estimators = 100)

#训练模型
m1.fit(x,y)
m2.fit(x,y)
m3.fit(x,y)

#应用模型
pre_1 = m1.predict(x)
pre_2 = m2.predict(x)
pre_3 = m3.predict(x)

#计算正确率
r1 = pre_1 == y
r2 = pre_2 == y
r3 = pre_3 == y
print("DecisionTreeClassifier's 正确率为：%.2f%%"%np.mean(r1*100))
print("AdaBoostClassifier's 正确率为%.2f%%"%np.mean(r2*100))
print("GradientBoostingClassifier's 正确率为:%.2f%%"%np.mean(r3*100))
