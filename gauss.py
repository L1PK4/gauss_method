import numpy as np

a = [[1, 2, 3], [5, -3, 4], [12, 3, 3]]
b = [3, 4, 1]

def col_plus_col(A, i, j, coef):
	A[j] = [A[j][k] + coef * A[i][k] for k in range(len(A[j]))]
	# for k in range(len(A[j])):
	# 	A[j][k] += coef * A[i][k]

def max_abs_iter(a):
	m = np.fabs(a[0])
	i = 0
	for j in range(len(a)):
		if m <= np.fabs(a[j]):
			m = np.fabs(a[j])
			i = j
	return i

def gauss_method(A, b):
	n = len(b)
	for i in range(n - 1):
		arr = [A[k][i] for k in range(i, n)]
		# print(arr)
		maxi = max_abs_iter(arr)
		print(A)
		if i != maxi: 
			A[i], A[maxi] = A[maxi], A[i]
		# print(A)
		for j in range(i + 1, n):
			c = - A[j][i] / A[i][i]
			col_plus_col(A, i, j, c)
			b[j] += b[i] * c
	# print(A)
	for j in range(n - 1, 0, -1):
		for i in range(j - 1, -1, -1):
			c = - A[i][j] / A[j][j]
			A[i][j] = 0
			b[i] += c * b[j]
	# print(A)