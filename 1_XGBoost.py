import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import datasets

#参数字典
'''
params={
'booster':'gbtree',
'objective': 'multi:softmax', #多分类的问题
'num_class':10, # 类别数，与 multisoftmax 并用
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':12, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言,假设 h 在 0.01 附近，
min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, # 如同学习率
'seed':1000,
'nthread':7,# cpu 线程数
#'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 5000 # 迭代次数
'''
#训练集测试集
iris = datasets.load_iris()
x = iris.data[:,:2]
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=1)
data_train = xgb.DMatrix(x_train,label=y_train)
data_test = xgb.DMatrix(x_test,label=y_test)

#参数
params = {}
params['objective'] = 'multi:softmax'#多分类
params['num_class'] = 3#类别数
params['max_depth'] = 6#最大树深，越大越容易过你和
params['silent'] = 1#打印
params['eta'] = 0.1#如同学习率
params['nthread'] = 4#线程数
num_round = 10
watchlist = [(data_train,'train'),(data_test,'test')]

#训练与预测
model = xgb.train(params,data_train,num_round,watchlist)
pre = model.predict(data_test)
print('error:%.3f%%'%(sum(int(pre[i])!=y_test[i] for i in range(len(y_test)))/float(len(y_test))))