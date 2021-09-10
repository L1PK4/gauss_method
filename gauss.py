from functools import reduce
import numpy as np
from numpy import abs
from random import uniform, randint

EPS = 1e-3

def col_plus_col(A, i, j, coef):
	A[j] = [A[j][k] + coef * A[i][k] for k in range(len(A[j]))]

def swap(A, i1, i2):
	permut = list(range(len(A)))
	permut[i1] = i2
	permut[i2] = i1
	return A[permut, :]

def gauss_method_inverse(A):
	n = len(A)
	A = np.concatenate((A, np.eye(n)), axis=1)
	for i in range(n):
		maxi = i
		for tempi in range(i, n):
			if abs(A[tempi][i]) > abs(A[maxi][i]):
				maxi = tempi
		if abs(A[maxi][i]) < EPS:
			return None
		A = swap(A,i, maxi)
		A[i] /= A[i][i]
		for j in range(n):
			if j != i:
				col_plus_col(A, i, j, - A[j][i])
	return A[:, n::]

def iter(x0, A):
	x = x0
	while True:
		x = np.matmul(x, 2 * np.eye(len(x)) - np.matmul(A, x))
		yield x

def add_err(A):
	n = len(A)
	num = randint(n, n*n)
	for _ in range(num):
		A[randint(0, n - 1)][randint(0, n - 1)] += uniform(-10 * EPS, 10 * EPS)

def error(X, A):
	E = np.matmul(X, A) - np.eye(len(A))
	return reduce(lambda a, b : a + abs(b), np.reshape(E, len(A) * len(A)), 0)

def main():
	a = [[1, 2, 6], [-4, 4, 3], [9, -6, 2]]
	print(f"Заданная матрица:\n{np.array(a)}")
	A = gauss_method_inverse(a)
	print(f"Моя обратная\n{A}\nКомпьютерная\n{np.linalg.inv(a)}")
	add_err(A)
	print(f"Испорченная\n{A}")
	corrector = iter(A, a)
	n = int(input("Введите кол-во итераций: "))
	for i in range(1, n):
		X = next(corrector)
		print(f"Итерация {i}:\n{X}\n\tПогрешность = {error(X, a):.2}")

	

if __name__ == "__main__":
	main()