from numpy import *
n = 100

dataidx = list(range(n))
for i in range(n):
	randidx = int(random.uniform(0,len(dataidx)))
	print("%s,%s"%(i,randidx))
	del(dataidx[randidx])
#print(dataidx)