import numpy as np

a = [[1, 2, 3], [5, -3, 4], [12, 3, 3]]
b = [3, 4, 1]

def col_plus_col(A, i, j, coef):
	A[j] = [A[j][k] + coef * A[i][k] for k in range(len(A[j]))]

def max_abs_iter(a):
	m = np.fabs(a[0])
	i = 0
	for j in range(len(a)):
		if m <= np.fabs(a[j]):
			m = np.fabs(a[j])
			i = j
	return i

def gauss_method(a, b, inverse=False):
	n = len(b)
	A = a.copy()
	B = b.copy()
	if inverse:
		inv = np.eye(n, n)
	for i in range(n - 1):
		maxi = max_abs_iter([A[k][i] for k in range(i, n)])
		if A[i][maxi] == 0:
			return None
		if i != maxi: 
			A[i], A[maxi] = A[maxi], A[i]
			B[i], B[maxi] = B[maxi], B[i]
			if inverse:
				inv[i], inv[maxi] = inv[maxi], inv[i]
		for j in range(i + 1, n):
			c = - A[j][i] / A[i][i]
			col_plus_col(A, i, j, c)
			if inverse:
				col_plus_col(inv, i, j, c)
			B[j] += B[i] * c
	for j in range(n - 1, 0, -1):
		for i in range(j - 1, -1, -1):
			c = - A[i][j] / A[j][j]
			col_plus_col(A, j, i, c)
			if inverse:
				col_plus_col(inv, j, i, c)
			B[i] += c * B[j]
	if inverse:
		return inv
	else:
		return A, B