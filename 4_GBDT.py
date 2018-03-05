import numpy as np
#GDBT选择特征的过程，即构建分类回归树的过程
def findSplit(x,y):
	#x为训练数据，y为标签，x[][j]为训练数据的第j个特征，x[i]为第i个训练数据
	#minLoss 最小损失
	minLoss = np.iinfo(np.int32).max
	#是训练数据的第几维特征
	feature = 0
	#split表示切分值
	split = 0

	#遍历每个特征
	for j in range(0,len(x[0])):
		for i in range(0,len(x)):
			#c为第j个特征的逐个取值
			c = x[i][j]
			L = 0
			r1 = []
			r2 = []
			y1 = 0
			y2 = 0
			count = 0
			#划分
			for m in range(0,len(x)):
				if(x[m][j] > c):
					r1.append(x[m])
					y1+=y[m]
					count += 1
				else:
					r2.append(x[m])
					y2+=y[m]
			if(count == 0):
				y1 = 0
				y2 = y2/len(x)
			else:
				y1 = y1/count
				y2 = y2/(len(x)-count)
			#计算loss
			for t in range(0,len(x)):
				if(x[t] in r1):
					L+= (y[t]-int(y1))^2
				else:
					L+= (y[t]-int(y2))^2
			if L < minLoss:
				minLoss = L
				feature = j
				split = c
	return minLoss,feature,split

def testSplit():
	x = [[1, 0, 3], [1, 1, 3], [4, 0, 6]]
	y = [0, 1, 0]
	minLoss, feature, split = findSplit(x, y)
	print("minLoss:%s,feature:%s,split:%s" % (minLoss, feature, split))

#多分类特征选择过程,这里index来表示第几类
'''这是一个有6个样本的三分类问题。我们需要根据这个花的花萼长度，花萼宽度，花瓣长度，花瓣宽度
来判断这个花属于山鸢尾，杂色鸢尾，还是维吉尼亚鸢尾。具体应用到gbdt多分类算法上面。
我们用一个三维向量来标志样本的label。[1,0,0] 表示样本属于山鸢尾，[0,1,0] 表示样本属于杂色鸢尾，[0,0,1] 表示属于维吉尼亚鸢尾。
gbdt 的多分类是针对每个类都独立训练一个 CART Tree。所以这里，我们将针对山鸢尾类别训练一个 CART Tree 1。
杂色鸢尾训练一个 CART Tree 2 。维吉尼亚鸢尾训练一个CART Tree 3，这三个树相互独立。
http://www.cnblogs.com/ModifyRong/p/7744987.html'''
def findSplitForMulti(train_data,label_data,index):
	#样本数
	sample_num = len(label_data)
	#特征数
	feature_num = len(train_data[0])
	#当前类标签
	curent_label = []
	#最小loss初始化
	minloss = 1000000
	#第几个特征
	best_feature = 0
	#切割点值
	split = 0

	#获取当前类标签
	for label in label_data:
		curent_label.append(label[index])

	#遍历所有特征
	for feature in range(0,feature_num):
		#当前特征的所有特征值
		curent_value = []
		for idx in range(0,sample_num):
			curent_value.append(train_data[idx][feature])
		L = 0
		#对每个特征值，计算其作为分割点的loss
		for i in range(0,len(curent_value)):
			r1 = []
			r2 = []
			y1 = 0
			y2 = 0
			count = 0
			#计算每个样本在该分割点下的loss
			for j in range(0,sample_num):
				if(train_data[j][feature]<curent_value[i]):
					r1.append(train_data[j])
					y1+=curent_label[j]
					count+=1
				else:
					r2.append(train_data[j])
					y2+=curent_label[j]
			#计算y1,y2
			if(count == 0):
				y1 = 0
				y2 = y2 / sample_num
			else:
				y1 = y1/count
				y2 = y2/(sample_num-count)
			for m in range(0,sample_num):
				#print(curent_label[m])
				if train_data[m] in r1:
					L+= pow(curent_label[m]-y1,2)
				else:
					L+=pow(curent_label[m]-y2,2)
			#print(L)
			if L < minloss:
				minloss = L
				best_feature = feature
				split = curent_value[i]
	return minloss,best_feature,split

def testSplitForMulti():
	train_data = [[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9]]
	label_data =[[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
	minloss,best_feature,split = findSplitForMulti(train_data,label_data,0)
	print("minloss:%.2f,best_feature:%s,split_value:%s"%(minloss,best_feature,split))

testSplitForMulti()











