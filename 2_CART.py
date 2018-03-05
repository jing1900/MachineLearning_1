'''
CART 主要包括特征选择、回归树的生成、剪枝三部分







二、回归树的生成函数 createTree
输入：数据集
输出：生成回归树
1、得到当前数据集的最佳划分特征、最佳划分特征值
2、若返回的最佳特征为空，则返回最佳划分特征值（作为叶子节点）
3、声明一个字典，用于保存当前的最佳划分特征、最佳划分特征值
4、执行二元切分；根据最佳划分特征、最佳划分特征值，将当前的数据划分为两部分
5、在左子树中调用createTree 函数, 在右子树调用createTree 函数。
6、返回树。

注：在生成的回归树模型中，划分特征、特征值、左节点、右节点均有相应的关键词对应。

三、（后）剪枝：（CART 树一定是二叉树，所以，如果发生剪枝，肯定是将两个叶子节点合并）

输入：树、测试集
输出：树

1、判断测试集是否为空，是：对树进行塌陷处理
2、判断树的左右分支是否为树结构，是：根据树当前的特征值、划分值将测试集分为Lset、Rset两个集合；
3、判断树的左分支是否是树结构：是：在该子集递归调用剪枝过程；
4、判断树的右分支是否是树结构：是：在该子集递归调用剪枝过程；
5、判断当前树结构的两个节点是否为叶子节点：
是:
a、根据当前树结构，测试集划分为Lset,Rset两部分；
b、计算没有合并时的总方差NoMergeError，即：测试集在Lset 和 Rset 的总方差之和；
c、合并后，取叶子节点值为原左右叶子结点的均值。求取测试集在该节点处的总方差MergeError，；
d、比较合并前后总方差的大小；若NoMergeError > MergeError,返回合并后的节点；否则，返回原来的树结构；
否：
返回树结构。
http://blog.csdn.net/qq_32933503/article/details/78408259

http://blog.csdn.net/xiaoxiaowenqiang/article/details/77119364

https://github.com/Ewenwan/PyML/tree/master/8RegressionTree
'''
#-*- coding:utf-8 -*-
import numpy as np
from numpy import *
import os

import sys
sys.setrecursionlimit(1000000)
#加载数据集
def loadDataSet(fileName):      #将一个文本文件导入到列表中
	dataMat = []                #创建一个dataMat空列表
	fr = open(fileName)
	for line in fr.readlines():  #一行一行的读取文本文件
		curLine = line.strip().split('\t')
		fltLine = map(float,curLine) #将文本文件转化为字符型
		#print(list(fltLine))
		dataMat.append(fltLine)
	return dataMat

def loadData(filename):
	data = loadtxt(filename)
	return data
# 特征选择：输入：       输出：最佳特征、最佳划分值
'''
1、选择标准
遍历所有特征Fi：遍历每个特征的所有特征值Zi；找到Zi，使划分后的总方差最小

2，停止划分的条件：
	1，当前数据集中的标签相同，返回当前的标签
	2，划分前后的总方差差距很小，数据不划分，返回的划分属性为空，返回的最佳划分点为当前所有标签的均值
	3，划分后的左右两个数据集的样本数量很小，返回的划分属性为空，返回的最佳划分点为当前所有标签的均值

当划分的数据满足上述条件之一，返回的最佳划分值作为叶子节点
当划分后的数据不满足上述要求时，找到最佳划分的属性，及最佳划分特征值

'''
#计算总的方差
def calAllVar(dataSet):
	'''
	var求方差，即各项-均值 的平方 求和后 再除以N

	# 构建测试数据，均值为10
	sc = [9.7, 10, 10.3, 9.7, 10, 10.3, 9.7, 10, 10.3]

	# 输出var, 即(0.09 + 0 + 0.09 + 0.09 + 0 + 0.09 + 0.09 + 0 + 0.09) = 0.54, 再0.54 / 9=0.06, 输出0.06
	print(np.var(sc))
	'''

	return var(dataSet[:,-1])*shape(dataSet)[0]#这里计算的是总方差，因此还要乘上数据行数

#根据给定的特征和特征值划分数据集
def dataSplit(dataSet,feature,featnumber):
	'''nonzero(a)返回数组a中值不为零的元素的下标，它的返回值是一个长度为a.ndim(数组a的轴数)的元组，
	# 元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值
	#b1 = np.array([True, False, True, False])
	#>> > np.nonzero(b1)
	#(array([0, 2]),)
	# 获取到dataset中特征列的值大于划分值的部分de行号，因是一维数组，这里取元祖的第1维元素'''
	dataL,dataR = [],[]
	#print(np.where(dataSet[:, feature] > featnumber))
	dataL = dataSet[np.where(dataSet[:,feature]>featnumber)[0],:]
	#dataL = dataSet[nonzero(dataSet[:,feature] > featnumber)[0],:]
	#dataR = dataSet[nonzero(dataSet[:,feature]<=featnumber)[0],:]
	dataR = dataSet[np.where(dataSet[:, feature] <= featnumber)[0], :]
	#print(shape(dataSet),shape(dataL),shape(dataR))
	#print(shape(dataL))

	return dataL,dataR

#特征选择,
'''
输入：
	数据集、op = [m,n],op里的参数依次表示剪枝前总方差与剪枝后总方差差值的最小值，和数据集划分为左右两个子数据集后，子数据集中的样本的最少数量；

输出：
	最佳特征、最佳划分特征值
	遇到停止条件，则输出当前数据集的标签平均值作为叶子节点值

过程：
	1、判断数据集中所有的样本标签是否相同，是：返回当前标签；
	2、遍历所有的样本特征，遍历每一个特征的特征值。计算出每一个特征值下的数据总方差，找出使总方差最小的特征、特征值
	3、比较划分前和划分后的总方差大小；若划分后总方差减少较小，则返回的最佳特征为空，返回的最佳划分特征值会为当前数据集标签的平均值。
	4、比较划分后的左右分支数据集样本中的数量，若某一分支数据集中样本少于指定数量op[1]，则返回的最佳特征为空，
	返回的最佳划分特征值会为当前数据集标签的平均值。
	5、否则，返回使总方差最小的特征、特征值
'''
def chooseBestFeature(dataSet,op=[1,4]):

	'''停止条件1，判断数据集中所有的样本标签是否相同，是：返回当前标签的均值，作为叶子节点值；'''
	if((len(set(dataSet[:,-1].T.tolist()[0])))==1):
		regleaf = mean(dataSet[:,-1])
		return None,regleaf

	'''遍历所有的样本特征，遍历每一个特征的特征值。计算出每一个特征值下的数据总方差，找出使总方差最小的特征、特征值'''
	m,n = shape(dataSet)#m个样本，n-1个特征
	Serror = calAllVar(dataSet)#总方差
	bestFeture = -1;bestnum = 0;lowError= inf#初始化最佳特征，最佳特征值和最小误差
	for f in range(0,n-1):	# 遍历所有特征
		for n in set(dataSet[:,f].T.tolist()[0]):#和其特征值
			dataL,dataR = dataSplit(dataSet,f,n)#根据特征和特征值将数据划分为左右两个数据子集
			#如果划分后的数据集数据甚少，直接continue
			if(shape(dataL)[0]<op[1] or shape(dataR)[0]<op[1]):
				#print(shape(dataL),shape(dataR))
				continue
			tmperror = calAllVar(dataL)+calAllVar(dataR)
			#print(tmperror)
			if(tmperror < lowError):
				#print(f,Serror,lowError)
				lowError = tmperror
				bestFeture = f
				bestnum = n
				#print(f,Serror,lowError)
	'''停止条件2：如果划分前后方差相差不大,则停止划分'''
	if(Serror - lowError < op[0]):
		#print('Serror 2')
		return None,mean(dataSet[:,-1])
	#否则，则将其按最优划分点划分
	dataL,dataR = dataSplit(dataSet,bestFeture,bestnum)
	'''停止条件3：比较划分后的左右分支数据集样本中的数量，若某一分支数据集中样本少于指定数量op[1]，则返回的最佳特征为空'''
	if(shape(dataL)[0] <op[1] or shape(dataR)[0]<op[1]):
		#print('Serror 3')
		return None,mean(dataSet[:-1])
	#print(bestFeture,bestnum)
	return bestFeture,bestnum

#生成决策树,在生成的回归树模型中，划分特征、特征值、左节点、右节点均有相应的关键词对应。
def createTree(dataSet,op=[1,4]):
	bestFeat,bestNum = chooseBestFeature(dataSet,op)#获取最佳分割点
	#print(bestFeat,bestNum)
	#满足三个终止条件的情况，返回叶子节点的值，即label
	if(bestFeat == None):
		return bestNum
	#构建决策树字典
	regTree = {}
	regTree['spidx'] = bestFeat
	regTree['spval'] = bestNum
	#获取左右集合
	dataL,dataR = dataSplit(dataSet,bestFeat,bestNum)
	#递归生成左右子树
	regTree['left'] =createTree(dataL,op)
	regTree['right'] = createTree(dataR,op)
	return regTree


#后剪枝操作
#判断所给节点是否为叶子节点
def istree(tree):
	#True:树，False:叶子节点
	return (type(tree).__name__ == 'dict')

## 递归的计算两个叶子的均值，从而返回树的平均值  塌陷处理
def getMean(tree):
	if istree(tree['left']): tree['left'] = getMean(tree['left'])
	if istree(tree['right']): tree['right'] = getMean(tree['right'])
	return (tree['left']+tree['right'])/2.0

#后剪枝 （CART 树一定是二叉树，所以，如果发生剪枝，肯定是将两个叶子节点合并）
'''1、判断测试集是否为空，是：对树进行塌陷(合并）处理
2、判断树的左右分支是否为树结构，是：根据树当前的特征值、划分值将测试集分为Lset、Rset两个集合；
3、判断树的左分支是否是树结构：是：在该子集递归调用剪枝过程；
4、判断树的右分支是否是树结构：是：在该子集递归调用剪枝过程；
5、判断当前树结构的两个节点是否为叶子节点：
是:
a、根据当前树结构，测试集划分为Lset,Rset两部分；
b、计算没有合并时的总方差NoMergeError，即：测试集在Lset 和 Rset 的总方差之和；
c、合并后，取叶子节点值为原左右叶子结点的均值。求取测试集在该节点处的总方差MergeError，；
d、比较合并前后总方差的大小；若NoMergeError > MergeError,返回合并后的节点；否则，返回原来的树结构；
否：
返回树结构。
'''
def prunetree(tree,testset):
	#1、判断测试集是否为空，是：对树进行塌陷(合并）处理,直接返回树的均值
	if(shape(testset)[0] == 0):
		return getMean(tree)
		#判断树的左右分支是否为树结构，是：根据树当前的特征值、划分值将测试集分为Lset、Rset两个集合；

	if(istree(tree)):

		keys = list(tree.keys())
		if 'left' in keys or 'right' in keys:
			Lset,Rset = dataSplit(testset,tree['spidx'],tree['spval'])
		#判断树的左右分支是否为树结构，若是，则在该子集递归调用剪枝过程
		if(istree(tree['left'])):
			tree['left'] = prunetree(tree['left'],Lset)
		if(istree(tree['right'])):
			tree['right'] = prunetree(tree['right'],Rset)
	else:
		return getMean(tree)
	#判断当前树结构的两个节点是否为叶子节点，
	if istree(tree):
		if not istree(tree['left']) and not istree(tree['right']):
			dataL,dataR = dataSplit(testset,tree['spidx'],tree['spval'])
			#计算没有合并时的总方差，即在左右测试集上的总方差之和
			#power(x1,2),对x1中的每个元素都求2次方,这里因是叶子节点，tree['left’】是唯一的,计算划分到左子集的每个测试集与其之差的平方
			NoMergeError = sum(power(dataL[:,-1]-tree['left'],2))+ sum(power(dataR[:,-1]-tree['right'],2))

			#合并后，叶子节点为原左右节点的均值
			leafval = getMean(tree)
			MergeError = sum(power(testset[:,-1]-leafval,2))
			#比较和并前后总方差大小，若不融合的方差大，则返回合并后的节点，否则返回原来树结构
			if NoMergeError > MergeError:
				print("the leaf merge")
				return leafval
			else:
				return tree
		else:
			return tree
	else:
		return getMean(tree)



#预测单个样本
def forecastsample(tree,testset):
	#如果只有一个叶子节点，返回该值作为分类的标签值
	if not istree(tree):
		return float(tree)
	if(testset[0,tree['spidx']] > tree['spval']):
		#print('2')
		if istree(tree['left']):
			return forecastsample(tree['left'],testset)
		else:
			return float(tree['left'])
	else:
		if istree(tree['right']):
			return forecastsample(tree['right'],testset)
		else:
			return float(tree['right'])



#预测整个数据集
def forecast(tree,testset):
	m = shape(testset)[0]
	pre_y = mat(zeros((m,1)))
	for i in range(0,m):
		#print(forecastsample(tree,testset[i]))
		pre_y[i,0]=forecastsample(tree,testset[i])
	return pre_y

if __name__ == "__main__":
	#创建树
	datamat = loadData("data/ex2.txt")
	data = mat(datamat)
	op = [1,6]
	tree = createTree(data,op)

	#测试
	testset = loadData("data/ex2test.txt")
	test = mat(testset)
	#树剪枝操作
	tree = prunetree(tree,test)
	y = test[:,-1]
	pre_y = forecast(tree,test)
	#计算相关系数
	'''corrcoef(x,y)表示序列x和序列y的相关系数,得到的结果是一个2*2矩阵,其中对角线上的元素分别表示x和y的自相关,
	非对角线上的元素分别表示x与y的相关系数,故这里取相关矩阵的非对角线上元素'''
	print(corrcoef(pre_y,y,rowvar=0)[0,1])