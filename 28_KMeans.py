# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from numpy import *

class KMeans(object):
	def __init__(self):
		pass

	def load_data(self,filename):
		dataset = []
		file = open(filename,'r')
		for line in file.readlines():
			linearr = line.strip().split('\t')
			dataset.append([float(linearr[0]),float(linearr[1])])
		return dataset

	#计算距离平方,d = sqrt( (v1(i)-v2(i)^2+(v1(j)-v2(j)^2 ),这里没有进行开方，为了避免不必要的误差
	def cal_distance(self,v1,v2):
		return sum(power(v1-v2,2))

	#用随机样本初始化质心,输入：数据集和聚类数k
	#这里随机初始化质心要去重，在样本点比较少且k比较小的情况下，不去重会因选重（chong）点而少分一类
	def init_centroids(self,dataset,k):
		n_samples,dim = shape(dataset)
		centoids = zeros((k+1,dim))#这里形状用小括号围起来

		s = set()#set,集合，无序，但里面的内容不能重复
		for i in range(1,k+1):
			while True:#这里巧妙利用了一个while，使其能一直选数，直到选到不重复的随机数放入集合
				index = int(random.uniform(0,n_samples))#用random.uniform()来选样本大小内的随机数
				if index not in s:
					s.add(index)#集合添加元素用add
					break
			centoids[i,:] = dataset[index,:]

		return centoids

	#计算当前分簇的cost,输入：clusters（两列）；第一列存这个样本点属于哪个簇 ，第二列存这个样本点和样本中心的误差
	def cal_cost(self,clusters):
		len = clusters.shape[0]#多少个样本点
		sum = 0.0
		for i in range(len):
			sum = sum+clusters[i,1]
		return sum

	#kmeans算法
	def kmeans(self,dataset,k):
		n_samples = dataset.shape[0]#样本数
		clusters = mat(zeros((n_samples,2)))##两列，第一列存样本属于哪个簇，第二列存样本和簇中心的误差
		for i in range(n_samples):
			clusters[i,0] = -1#簇号初始化为-1
		clusterchanged = True#标记位,簇是否还需要划分

		#初始化质心
		centoids = self.init_centroids(dataset,k)

		#收敛后，clusterchange置为false
		while clusterchanged:
			clusterchanged = False
			#对每个样本点
			for i in range(n_samples):
				#对每个质心，计算距离
				min_dis = 1000000.0
				min_index = 0 #最小距离的质心
				#找到离得最近的质心
				for j in range(1,k+1):
					dis = self.cal_distance(dataset[i,:],centoids[j,:])
					if dis < min_dis:
						min_dis = dis
						min_index = j
				#更新分簇关系
				if clusters[i,0] != min_index:
					#只有迭代到所有点的质心不变，方可结束
					clusterchanged = True
					clusters[i,:] = min_index,min_dis
				else:
					clusters[i,1] = min_dis
			#更新簇中心
			for j in range(1,k+1):
				clusterpoints = dataset[nonzero(clusters[:,0].A == j)[0]]#找到clusters中簇为j的行坐标，进而从dataset中得到这些值
				centoids[j,:] = mean(clusterpoints,axis=0)#axis = 0,按列取平均，进而求得上述簇点的质心

		return centoids,clusters

	#画出聚类结果图：
	def show_clusters(self,dataset,k,centoids,clusters):
		n_samples,dim = shape(dataset)
		if dim > 2:
			print("sorry, we can only show 2d plot")
			return 1
		mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']#非中心样本点的颜色标记
		if k > len(mark):
			print("sorry, your k is too large")
			return 1
		#绘制所有非中心样本点
		for i in range(n_samples):
			markindex = int(clusters[i,0])#该样本簇号
			plt.plot(dataset[i,0],dataset[i,1],mark[markindex-1])
		mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']#中心点颜色标记
		for i in range(1,k+1):
			plt.plot(centoids[i,0],centoids[i,1],mark[i-1],markersize = 12)
		plt.title("k=%s"%k)
		plt.show()

	def show_cost(self,ks,costs):

		plt.plot(ks,costs,'b--',linewidth= 1)
		plt.xlabel("k")
		plt.ylabel('cost')
		plt.show()


if __name__ == "__main__":
	k = KMeans()
	#加载数据集
	dataset = k.load_data('data/28_test.txt')
	dataset = mat(dataset)
	#聚类
	ks= [2,3,4,5,6,7,8]
	costs = []
	for i in range(len(ks)):
		centoids,clusters = k.kmeans(dataset,ks[i])

		#计算cost
		cost = k.cal_cost(clusters)
		costs.append(cost)
		#绘聚类结果图
		k.show_clusters(dataset,ks[i],centoids,clusters)
	#绘制不同k下cost变化情况
	k.show_cost(ks,costs)





