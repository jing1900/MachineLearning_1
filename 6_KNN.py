'''KNN : 以距离最近的k个已知类别的点的类别，来多数投票的原则得到新样本点的类别'''
# -*- coding:utf-8 -*-

import csv
import operator
import math
import random

class KNearestNeighbor(object):
	def __init__(self):
		pass

	#加载数据
	def loaddata(self,filename,split,trainingset,testset):
		with open(filename,'r') as file:
			lines = csv.reader(file)#读取所有行
			dataset = list(lines)#转为列表
			'''
			range(start, stop[, step])
			start: 计数从 start 开始。默认是从 0 开始。例如range（5）等价于range（0， 5）;
			stop: 计数到 stop 结束，但不包括 stop。例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
			step：步长，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)
			'''
			for i in range(len(dataset)):
				for j in range(4):
					dataset[i][j] = float(dataset[i][j])
				#这里split为0.75,每个数据有0.75的可能被划分为训练集
				if random.random() < split:
					trainingset.append(dataset[i])
				else:
					testset.append(dataset[i])

	#计算距离，dim表示有几维
	def caldistance(self,traininstance,testinstance,dim):
		distance = 0
		for i in range(dim):
			distance += pow(traininstance[i]-testinstance[i],2)
		return math.sqrt(distance)

	#返回距某测试实例最近的k个距离
	def getknearest(self,trainset,testinstance,k):
		#计算所有训练实例距其的距离
		distances = []
		#特征数，即计算距离的维度,测试实例的最后一维为标签
		dim = len(testinstance) -1
		for i in range(len(trainset)):
			dist = self.caldistance(trainset[i],testinstance,dim)
			distances.append((trainset[i],dist))
		#按距离从小到大排序，sort默认是升序，由小到大
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for i in range(k):
			#这里要取排序后的distances的前第一维数据（样本本身），因为第二维是距离
			neighbors.append(distances[i][0])
		return neighbors

	#计算投票，按少数服从多数的原则
	def getresponse(self,neighnors):
		votes = {}
		#计票
		for i in range(len(neighnors)):
			if(neighnors[i][-1] in votes):
				votes[neighnors[i][-1]] +=1
			else:
				votes[neighnors[i][-1]] = 1
		#排序,按计票数从大往小排
		'''
		sort 与 sorted 区别：
		sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
		list 的 sort 方法返回的是对已经存在的列表进行操作，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。'''
		sortedvotes = sorted(votes.items(),key=operator.itemgetter(1),reverse=True)
		#返回排序后的第一行的第一个值，即类别
		return sortedvotes[0][0]

	#计算准确率,predictions为预测结果
	def calaccuracy(self,testset,predictions):
		accuracy = 0
		for i in range(len(testset)):
			if(testset[i][-1] == predictions[i]):
				accuracy += 1
		return accuracy/float(len(testset))

	#run
	def run(self):
		#划分数据集
		trainingset,testset = [],[]
		split = 0.75
		self.loaddata("data/6knn_testdata.txt",split,trainingset,testset)
		#预测
		predictions = []
		k = 3
		for i in range(len(testset)):
			neighbors = self.getknearest(trainingset,testset[i],k)
			prediction = self.getresponse(neighbors)
			predictions.append(prediction)
		#计算精度
		accuracy = self.calaccuracy(testset,predictions)
		print("accuracy:",accuracy)

if __name__ == "__main__":
	k = KNearestNeighbor()
	k.run()