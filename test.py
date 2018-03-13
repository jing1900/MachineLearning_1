import tensorflow as tf
from sklearn.datasets import load_digits
import numpy as np

batch_size = 8 #这里使用的是MBGD
#加载数据集
digits = load_digits()
x = digits.data.astype(np.float32)
y = digits.target.astype(np.float32).reshape(-1,1)#转为任意行，一列的数据

#数据预处理
from sklearn.preprocessing import MinMaxScaler
x_data = MinMaxScaler().fit_transform(x)
from sklearn.preprocessing import OneHotEncoder
y_data = OneHotEncoder().fit_transform(y).todense()#转换为稠密矩阵
#将数据转为图像
x_data = x_data.reshape(-1,8,8,1)#batch,height,weight,channel

#生成每一个batch大小的数据
def generateBatch(x,y,n_samples,batch_size):
	for i in range(n_samples // batch_size):
		start = i*batch_size
		end = start+batch_size
		batch_x = x[start:end]
		batch_y = y[start:end]
		yield batch_x,batch_y

tf.reset_default_graph()#清除默认图的堆栈,并设置全局图为默认图
#输入层
tf_x = tf.placeholder(tf.float32,[None,8,8,1])#未知：None
tf_y = tf.placeholder(tf.float32,[None,10])

#卷积层1+激活层1，输出的shape为？,8,8,10,因为padding方式为same，输出 = 输入/stride
conv_filter_w1 = tf.Variable(tf.random_normal([3,3,1,10]))#kernel 长，宽，in_channel = 输入图层数,out_channel = kernel数
conv_filter_b1 = tf.Variable(tf.random_normal([10]))
conv_filter1_out = tf.nn.relu(tf.nn.conv2d(tf_x,conv_filter_w1,[1,1,1,1],padding='SAME')+conv_filter_b1)#这里不要忘了加偏置

#池化层1,该层输出的shape为?*4*4*10
max_pool1_out = tf.nn.max_pool(conv_filter1_out,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#卷基层2+激活层2
conv_filter_w2 = tf.Variable(tf.random_normal([3,3,10,5]))
conv_filter_b2 = tf.Variable(tf.random_normal([5]))
conv_filter2_out = tf.nn.relu(tf.nn.conv2d(max_pool1_out,conv_filter_w2,[1,2,2,1],padding='SAME')+conv_filter_b2)

#BN层+激活层
batch_mean,batch_val = tf.nn.moments(conv_filter2_out,axes=[0,1,2],keep_dims=True)
offset = tf.Variable(tf.zeros([5]))
scale = tf.Variable(tf.ones([5]))
variance_epsilon = 1e-3
bn_out = tf.nn.batch_normalization(conv_filter2_out,batch_mean,batch_val,offset=offset,scale=scale,variance_epsilon=variance_epsilon)
relu_bn_out = tf.nn.relu(bn_out)

#池化层2
max_pool2_out = tf.nn.max_pool(relu_bn_out,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#将特征图展开,用tf.reshape函数
max_pool2_out_flat = tf.reshape(max_pool2_out,[-1,1*1*5])

#全连接层+激活层，tf.matmul
fc_w = tf.Variable(tf.random_normal([5,50]))
fc_b = tf.Variable(tf.random_normal([50]))
fc_out = tf.nn.relu(tf.matmul(max_pool2_out_flat,fc_w)+fc_b)

#输出层+softmax,!!!注意该层为nn.softmax
out_w = tf.Variable(tf.random_normal([50,10]))
out_b = tf.Variable(tf.random_normal([10]))
pred = tf.nn.softmax(tf.matmul(fc_out,out_w)+out_b)

#计算交叉熵,tf.clip_by_value,用来将值框在一个范围
loss = -tf.reduce_mean(tf_y*tf.log(tf.clip_by_value(pred,1e-13,1.0)))
train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

#tf.arg_max(pred,1)是按行取最大值的下标
#tf.arg_max(tf_y,1)是按列取最大值的下标
y_pred = tf.argmax(pred,1)
bool_pred = tf.equal(y_pred,tf.argmax(tf_y,1))

#计算准确率
#先将correct_pred中数据格式转换为float32类型
#求correct_pred中的平均值，因为correct_pred中除了0就是1，因此求平均值即为1的所占比例，即正确率
correct_pred = tf.reduce_mean(tf.cast(bool_pred,tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())#全局变量初始化
	for epoch in range(1000):
		for batch_x,batch_y in generateBatch(x_data,y_data,y_data.shape[0],batch_size):
			sess.run(train_step,feed_dict={tf_x:batch_x,tf_y:batch_y})
		if(epoch % 100 == 0):
			correct = sess.run(correct_pred,feed_dict={tf_x:x_data,tf_y:y_data})#由于bn层存在，现在只支持批量的
			print(epoch,correct)
	#eval(),将其内的字符串当做合法语法处理，flatten（），将多维转为1维
	ypred = y_pred.eval(feed_dict={tf_x:x_data,tf_y:y_data}).flatten()
	print(ypred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y,ypred.reshape(-1,1))
print(accuracy)


