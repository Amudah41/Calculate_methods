#вариант №6
import copy
from enum import Enum
from functools import partial
import numpy as np
from numpy.linalg import det
from numpy.linalg import norm, solve

A = np.array([[12.785723, 1.534675, -3.947418], [1.534675, 9.709232, 0.918435], [-3.947418, 0.918435, 7.703946]])
b = np.array([9.60565, 7.30777, 4.21575])

def solve_gauss(A, b, epsilon=1e-8):
    n = A.shape[0]
    A = np.append(A, b[:, np.newaxis], axis=1)

    # прямой ход
    for k in range(n):
        tmp = A[k, k]
        if abs(tmp) < epsilon:
            raise ZeroDivisionError()
        for j in range(k, n+1):
            A[k, j] /= tmp
        for i in range(k + 1, n):
            tmp = A[i, k]
            for j in range(k, n+1):
                A[i, j] = A[i, j] - A[k, j] * tmp

    # обратный ход
    x = np.zeros(n)
    for i in range(n-1, 0-1, -1):
        x[i] = A[i, n]
        for j in range(i+1, n):
            x[i] -= A[i, j] * x[j]

    return x


def LU(A):
    n = A.shape[0]
    l = copy.deepcopy(A)
    u = copy.deepcopy(A)
    for i in range(n):
        for j in range(n):
            sum = 0.0
            for k in range(i):
                sum += l[j][k] * u[k][i]
            l[j][i] = A[j][i] - sum
            sum = 0.0
            for k in range(i):
                sum += l[i][k] * u[k][j]
            u[i][j] = (A[i][j] - sum)/l[i][i]
    return l, u

def choose_main_elements(A):
    n = A.shape[0]
    ord_x = np.arange(n)
    ord_b = np.arange(n)
    for k in range(n-1):
        i, j = np.unravel_index(np.abs(A[k:, k:]).argmax(), A[k:, k:].shape)
        i += k
        j += k

        tmp = copy.deepcopy(A[k, :])
        A[k, :], A[i, :] = A[i, :], tmp
        ord_b[k], ord_b[i] = ord_b[i], ord_b[k]

        tmp = copy.deepcopy(A[:, k])
        A[:, k], A[:, j] = A[:, j], tmp
        ord_x[k], ord_x[j] = ord_x[j], ord_x[k]

    return A, ord_x, ord_b



def solve_gauss_enhanced(A, b, epsilon=1e-8):

    def restore_order(a, ord_a):
        tmp = copy.deepcopy(a)
        for i in range(len(ord_a)):
            a[ord_a[i]] = tmp[i]
        return a

    A_, ord_x, ord_b = choose_main_elements(copy.deepcopy(A))
    x = solve_gauss(A_, b[ord_b], epsilon)
    return restore_order(x, ord_x)

def det_l(L):
    n = L.shape[0]
    d = 1.0
    for i in range(n):
        d*=L[i][i]
    return d

def addition(A, i, j):
    A_shortened = np.delete(A, i, 0)
    A_shortened = np.delete(A_shortened, j, 1)
    return (-1) ** (i + j) * det(A_shortened)


def invert(A):
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    A_inv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A_inv[i, j] = addition(A, i, j)
    return A_inv.transpose() / det(A)



x1 = solve_gauss(A, b)
x2 = solve_gauss_enhanced(A, b)
print("Невязка решения при обычном методе Гаусса: ", b - np.dot(A, x1))
print("Невязка решения при методе Гаусса с выбором главного элемента: ", b - np.dot(A, x2))
print()

L, U = LU(A)
x3=solve_gauss(U, solve_gauss(L, b))
print("Невязка решения при LU-разложении: ", b - np.dot(A, x2))
print()

C = copy.deepcopy(A)
C[0, 0] *= 1e-8
x1 = solve_gauss(C, b)
x2 = solve_gauss_enhanced(C, b)
print("Умножим первый элемент матрицы на 1e-8.")
print("Невязка решения при обычном методе Гаусса: ", b - np.dot(C, x1))
print("Невязка решения при методе Гаусса с выбором главного элемента: ", b - np.dot(C, x2))
print()

print("Обратная матрица: ")
print(invert(A))
print("Проверка: ")
print(np.dot(A, invert(A)))