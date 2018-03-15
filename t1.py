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
if __name__ == '__main__':
	str = sys.stdin.readline().strip()
	str = list(str)
	for i in range(len(str)):
		s = str[i]
		#如果在a-y，A-Y，ord 码+1
		if(ord('y') >= ord(s) >= ord('a') or ord('Y') >= ord(s) >= ord('A')):
			str[i] = chr(ord(s)+1)
		elif ord(s) == ord('z'):
			str[i] = chr(ord('a'))
		elif ord(s) == ord('Z'):
			str[i] = chr(ord('A'))
		else:
			str[i] = s
	print(''.join(str))

		#如果为 z，orz，变成97,（a，or
		#print(ord(str[i]))