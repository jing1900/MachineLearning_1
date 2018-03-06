'''逻辑回归：大致思想是由极大似然法，根据给定的样本值，得出模型最有可能的参数值，极大似然法的求解依赖梯度上升算法'''
from numpy import *
import matplotlib.pyplot as plt
import time

class LogisticRegression(object):
	def __init__(self):
		pass

	#加载数据
	def loaddata(self,filename):
		train_x = []
		train_y = []
		file = open(filename,'r')
		for line in file.readlines():
			lineArr = line.strip().split()
			train_x.append([1.0,float(lineArr[0]),float(lineArr[1])])
			train_y.append(float(lineArr[2]))
		return mat(train_x),mat(train_y).transpose()

	#sigmoid函数
	def sigmoid(self,x):
		return 1.0/(1+exp(-x))

	#训练
	'''
	输入：
		train_x:训练样本矩阵
		train_y:训练样本对应的标签
		ops：步长和迭代次数等参数
	'''
	def train(self,train_x,train_y,ops):
		starttime = time.time()#统计时间

		sample_num,feature_num = shape(train_x)#样本数和特征数
		alpha = ops['alpha']#一些参数
		max_iters = ops['iters']
		weights = ones((feature_num,1))#权重

		#通过随机梯度上升算法优化参数
		for i in range(max_iters):
			#梯度上升
			if ops['optimizeType'] == 'gradDescent':
				#这里由公式得到，具体见12题的参数更新公式
				outputs = self.sigmoid(train_x*weights)
				errors = train_y - outputs
				weights = weights + alpha * train_x.transpose()*errors
			#随机梯度上升
			elif ops['optimizeType'] == 'stocGradDescent':
				for j in range(sample_num):#比起批处理，这里是来一个更新一个，属于在线学习算法
					outputs = self.sigmoid(train_x[j,:]*weights)
					errors = train_y[j,0] - outputs
					weights = weights + alpha*train_x[j,:].transpose()*errors
			#改进的随机梯度上升，改进了alpha，随迭代次数增大而变小，另外随机的从数据中选取，避免周期性波动产生
			elif ops['optimizeType'] == 'smoothStocGradDescent':
				dataindex = list(range(sample_num))#得到样本编号列表
				for j in range(sample_num):
					alpha = 4.0/(1.0+i+j)+0.01
					#random.uniform(x, y),x -- 随机数的最小值，包含该值.   y -- 随机数的最大值，不包含该值。
					randidx =int(random.uniform(0,len(dataindex))) #从编号范围内随机获取idx
					outputs = self.sigmoid(train_x[randidx,:]*weights)
					errors = train_y[randidx,0]-outputs
					weights = weights + alpha*train_x[randidx,:].transpose()*errors
					del(dataindex[randidx])#这里可以这样做是因为，删了该处idx对应的元素，其他下标会往前补全，保证每个idx都会被随机访问一遍
			else:
				print('no support optimize type')
		print("train is complete, takes %s second"%(time.time() - starttime))
		return weights

	#预测,计算准确率
	def calaccuracy(self,weights,test_x,test_y):
		sample_nums = shape(test_x)[0]
		accuracy = 0.0
		for i in range(sample_nums):
			output = self.sigmoid(test_x[i,:]*weights)[0,0] > 0.5
			if output == bool(test_y[i,0]):
				accuracy += 1
		return float(accuracy)/sample_nums

	#绘图
	def show(self,train_x,train_y,weights):
		sample_nums,feature_nums = shape(train_x)
		#这里是因为train_x,第一维全被赋值1，后两维才是原来的两维，总共三维特征
		if feature_nums != 3:
			print("sorry, we can only show in 2d")
			return 1
		#画实例
		for i in range(sample_nums):
			if(int(train_y[i,0]) == 0):
				plt.plot(train_x[i,1],train_x[i,2],'or')
			elif(int(train_y[i,0]) == 1):
				plt.plot(train_x[i,1],train_x[i,2],'ob')
		#画分类线
		min_x1 = min(train_x[:,1])[0,0]
		max_x1 = max(train_x[:,1])[0,0]
		weights = weights.getA()#将矩阵转为数组
		#这里根据0 = w0x0 + w1x1 + w2x2，x0 = 1,解出x1，x2的相关关系
		min_x2 = float(-weights[0]-weights[1]*min_x1)/weights[2]
		max_x2 = float(-weights[0]-weights[1]*max_x1)/weights[2]
		plt.plot([min_x1,max_x1],[min_x2,max_x2],'-g')
		plt.xlabel("X1")
		plt.ylabel("X2")
		plt.show()

	#运行
	def run(self):
		#加载数据
		train_x,train_y = self.loaddata("data/12_testset.txt")
		test_x = train_x
		test_y = train_y

		#训练
		ops = {"alpha":0.01,"iters":200,"optimizeType":'stocGradDescent'}
		optimizeWeights = self.train(train_x,train_y,ops)

		#计算准确率
		accuracy = self.calaccuracy(optimizeWeights,test_x,test_y)
		print("the accuracy is %s",accuracy)
		#画图
		self.show(train_x,train_y,optimizeWeights)

if __name__ == "__main__":
	l = LogisticRegression()
	l.run()




