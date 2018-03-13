import tensorflow as tf
from sklearn.datasets import load_digits
import numpy as np

batch_size = 8#,使用MBGD，小批量梯度下降法，每次更新参数时，固定使用这么些个样本


#加载数据集
digits = load_digits()
x_data = digits.data.astype(np.float32)#shape：1797*64
y = digits.target.astype(np.float32).reshape(-1,1)#这里reshape(-1,1),将原有变成不知多少行，一列的数据，shape：1797*1


#数据预处理
#标准化x_data
from sklearn.preprocessing import MinMaxScaler#MinMaxScaler,x = (x-min)/(max - min),将属性缩放到一个指定的[0,1]之间，该方法可以维持稀疏矩阵中为0的条目
x_data = MinMaxScaler().fit_transform(x_data)
#对y_data 进行onehot编码
from sklearn.preprocessing import OneHotEncoder
y_data = OneHotEncoder().fit_transform(y).todense()#one_hot编码，todense：将稀疏矩阵转化为完整特征矩阵
#将输入转换为图片格式
x_data = x_data.reshape(-1,8,8,1)#batch ,height,weight,channel,-1表示未知，由其他参数自动计算

#生成每一个batch大小的样本
def generateBatch(X,Y,n_samples,batch_size):
	for i in range(n_samples // batch_size):
		start = i*batch_size
		end = start + batch_size
		batch_x = X[start:end]
		batch_y = Y[start:end]
		yield batch_x,batch_y #yield 的作用就是把一个函数变成一个 generator，调用带filed的函数会返回一个 iterable 对象

tf.reset_default_graph()#清除默认图的堆栈,并设置全局图为默认图
#输入层
tf_x = tf.placeholder(tf.float32,[None,8,8,1])#输入占位符
tf_y = tf.placeholder(tf.float32,[None,10])#输出占位符

#卷积层1+激活层1,relu_feature_maps1.shape=(?, 8, 8, 10),这是因为padd = same时，输出长宽 = 输入长宽/stride。
# 或者将same看做pad = 1，代入（w-f+2p）/s + 1
conv_filter_w1 = tf.Variable(tf.random_normal([3,3,1,10]))#filter长，宽，inchannel = 输入图层数，out_channel卷积核数
conv_filter_b1 = tf.Variable(tf.random_normal([10]))#形状为outchannel数
relu_feature_maps1 = tf.nn.relu(tf.nn.conv2d(tf_x,conv_filter_w1,strides = [1,1,1,1],padding='SAME')+conv_filter_b1)

#池化层1, max_pool1.shape=(?, 4, 4, 10)
max_pool1= tf.nn.max_pool(relu_feature_maps1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
#卷积层2+ 激活层2,relu_feature_maps2.shape = [?,2,2,5]
conv_filter_w2 = tf.Variable(tf.random_normal([3,3,10,5]))
conv_filter_b2 = tf.Variable(tf.random_normal([5]))
relu_feature_maps2 = tf.nn.relu(tf.nn.conv2d(max_pool1,conv_filter_w2,strides=[1,2,2,1],padding='SAME')+conv_filter_b2)

#BN归一化层和激活层,shape = (?,2,2,5)
batch_mean,batch_val = tf.nn.moments(relu_feature_maps2,axes=[0,1,2],keep_dims=True)
shift = tf.Variable(tf.zeros([5]))
scale = tf.Variable(tf.ones([5]))
variance_epsilon = 1e-3
bn_out = tf.nn.batch_normalization(relu_feature_maps2,batch_mean,batch_val,offset=shift,scale=scale,variance_epsilon=variance_epsilon)
relu_bn_maps = tf.nn.relu(bn_out)
#池化层2，shape = (?,1,1,5)
max_pool2 = tf.nn.max_pool(relu_bn_maps,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#将特征图展开,shape = (?,5)
max_pool2_flat = tf.reshape(max_pool2,[-1,1*1*5])

#全连接层
fc_w1 = tf.Variable(tf.random_normal([5,50]))
fc_b1 = tf.Variable(tf.random_normal([50]))
fc_out1 = tf.nn.relu(tf.matmul(max_pool2_flat,fc_w1)+fc_b1)#矩阵相乘操作

#输出层
out_w1 = tf.Variable(tf.random_normal([50,10]))
out_b1 = tf.Variable(tf.random_normal([10]))
pred = tf.nn.softmax(tf.matmul(fc_out1,out_w1)+out_b1)

#计算交叉熵，首先，用 tf.log 计算 y 的每个元素的对数。接下来，我们把 y_ 的每一个元素和 tf.log(y) 的对应元素相乘。
# 最后，用 tf.reduce_mean 计算熵的平均值
# （注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能。
#（tf.log:计算log，一个输入计算e的ln，两输入以第二输入为底）
# （tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。）
loss= -tf.reduce_mean(tf_y*tf.log(tf.clip_by_value(pred,1e-13,1.0)))
#训练步
train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

#tf.arg_max(pred,1)是按行取最大值的下标
#tf.arg_max(y,1)是按列取最大值的下标
y_pred = tf.argmax(pred,1)
bool_pred = tf.equal(y_pred,tf.argmax(tf_y,1))

#先将correct_pred中数据格式转换为float32类型
#求correct_pred中的平均值，因为correct_pred中除了0就是1，因此求平均值即为1的所占比例，即正确率
correct_pred = tf.reduce_mean(tf.cast(bool_pred,tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(1000):
		for batch_x,batch_y in generateBatch(x_data,y_data,y_data.shape[0],batch_size):
			sess.run(train_step,feed_dict={tf_x:batch_x,tf_y:batch_y})
		if(epoch % 100 == 0):
			correct = sess.run(correct_pred,feed_dict={tf_x:x_data,tf_y:y_data})
			print(epoch,correct)
	#eval(),将其内的字符串当做有效语法.flatten():将多维将为一维
	#需要注意的是，这个模型还不能用来预测单个样本，因为在进行BN层计算时，单个样本的均值和方差都为0，会得到相反的预测效果，解决方法详见归一化层。
	ypred = y_pred.eval(feed_dict={tf_x:x_data,tf_y:y_data}).flatten()

from sklearn.metrics import accuracy_score
score = accuracy_score(y,ypred.reshape(-1,1))
print(score)