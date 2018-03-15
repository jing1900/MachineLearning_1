'''
#coding=utf-8
import sys
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))
'''

'''
#coding=utf-8
import sys
if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    ans = 0
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))
        for v in values:
            ans += v
    print(ans)
'''
import sys
from operator import itemgetter, attrgetter
import numpy as np
if __name__ == "__main__":
	#获取N，M
	arr = sys.stdin.readline().split()
	N,M = int(arr[0]),int(arr[1])

	#最大木头长度
	max_wood = 1000
	#二维列表，共三列，每行为一个路径，每行的第一个值为start，第二个值为end，第三个值为该路径所需木头数
	mat = []

	#获取路径
	for i in range(M):
		line = sys.stdin.readline().strip()
		values = list(map(int,line.split()))
		mat.append(values)

	#标记已经能走到的路径
	mat.sort(key=itemgetter(0))
	print(mat)
	start,end,max_wood = mat[0][0],mat[0][1],mat[0][2]
	for i in range(1,len(mat)):
		if start <= mat[i][0] <= end  and mat[i][1] > end :
			end = mat[i][1]
			if mat[i][2] > max_wood:
				max_wood = mat[i][2]
		elif mat[i][0] < start and start<=mat[i][1] <=end:
			start = mat[i][0]
			if(mat[i][2] >max_wood):
				max_wood = mat[i][2]
		else:
			continue

	if(end == N and start == 1):
		print(max_wood)




