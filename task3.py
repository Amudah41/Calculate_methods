import numpy as np
from numpy.linalg import eig, norm
from modules import *
import copy
from numpy.linalg import det

A = np.array([[-0.95121, -0.09779, 0.35843],[ -0.09779, 0.61545, 0.022286],
              [0.35843, 0.02229, -0.95729]])

def Jacobi(A, eps):
    def rotate(A, V, ik, jk):
        n = A.shape[0]
        aDiv = A[ik][ik] - A[jk][jk]
        phi = 0.5 * np.arctan(2 * A[ik][jk] / aDiv)
        c = np.cos(phi)
        s = np.sin(phi)
        for i in range(n):
            if (i != ik) and (i != jk):
                A[ik][i] = c * A[i][ik] + s * A[i][jk]
                A[jk][i] = (-1) * s * A[i][ik] + c * A[i][jk]
                A[i][ik] = A[ik][i]
                A[i][jk] = A[jk][i]
        temp1 = (c ** 2) * A[ik][ik] + 2 * c * s * A[ik][jk] + (s ** 2) * A[jk][jk]
        temp2 = (s ** 2) * A[ik][ik] - 2 * c * s * A[ik][jk] + (c ** 2) * A[jk][jk]
        A[ik][ik] = temp1
        A[jk][jk] = temp2
        A[ik][jk] = 0.0
        A[jk][ik] = 0.0
        for i in range(n):
            temp1 = c * V[i][ik] + s * V[i][jk]
            temp2 = (-1) * s * V[i][ik] + c * V[i][jk]
            V[i][ik] = temp1
            V[i][jk] = temp2

    n = A.shape[0]
    V = np.identity(n) * 1.0
    def over_diagonal_argmax(A):
        n = A.shape[0]
        D = np.zeros_like(A)
        for i in range(n - 1):
            for j in range(i + 1, n):
                D[i][j] = A[i][j]
        ik, jk = np.unravel_index(np.abs(D).argmax(), D.shape)
        assert ik < jk
        return A[ik][jk], ik, jk

    current, ik, jk = over_diagonal_argmax(A)
    while np.abs(current) >= eps:
        rotate(A, V, ik, jk)
        current, ik, jk = over_diagonal_argmax(A)

    return np.diagonal(A), V


def power_method(A, eps, Yk=None):
    if Yk is None:
        Yk = np.array([-0.01, -0.01, -0.01])
    res = 1
    k = 0
    p = np.argmax(np.abs(Yk))
    while res >= eps:
        k += 1
        Yk = Yk / Yk[p]
        Yk_next = np.dot(A, Yk)
        l1 = Yk_next[p] / Yk[p]
        res = norm(np.dot(A, Yk_next) - l1 * Yk_next, np.inf)
        Yk = Yk_next
        if k >= 100:
            print("не сходится")
            return 0, 0, k
    return l1, Yk / norm(Yk), k


def scal_prod(A, eps, Yk=None):
    if Yk is None:
        Yk = np.array([-0.01, -0.01, -0.01])
    res = 1
    k = 0
    p = np.argmax(np.abs(Yk))
    while res >= eps:
        k += 1
        Yk = Yk / Yk[p]
        Yk_next = np.dot(A, Yk)
        l1 = np.dot(Yk_next, Yk) / np.dot(Yk, Yk)
        res = norm(np.dot(A, Yk_next) - l1 * Yk_next, np.inf)
        Yk = Yk_next
        if k >= 100:
            print("не сходится")
            return 0, 0, k
    return l1, Yk / norm(Yk), k


def spec_bound(A, eps):
    l1, l1_vec, _ = power_method(A, eps)
    B = A - l1 * np.identity(A.shape[0])
    lB, lB_vec, _ = power_method(B, eps)
    res_vec = (l1_vec + lB_vec) / norm(l1_vec + lB_vec)
    return lB + l1, res_vec

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


def wielandt_refinement(A, eps):
    lk = -1
    k = 0
    res = 1
    while res >= eps:
        k += 1
        W = A - lk * np.identity(A.shape[0])
        l, v, i = scal_prod(invert(W), eps)
        lk = 1 / l + lk
        res = norm(np.dot(A, v) - lk * v)
        if k >= 500:
            print("не сходится")
            return 0, 0, k
    return lk, v, k


C = copy.deepcopy(A)
l, x = Jacobi(C, 1e-6)
print("Метод Якоби ")
print("Собственные числа:", l)
print("Собственные векторы:\n", x)
print("Нормы векторов: ")
for i in range(len(x)):
    print(norm(x[:, i]))
print()

l, x, k = power_method(A, 1e-3)
print("Степенной метод")
print("Собственное число:", l)
print("Собственный вектор:", x)
print("Число итераций: ", k)
print("Норма вектора: ", norm(x))
print("Невязка: ", np.dot(A, x)-l*x)
print()


l, x, k = scal_prod(A, 1e-3)
print("Метод скалярных произведений")
print("Собственное число:", l)
print("Собственный вектор:", x)
print("Количество итераций: ", k)
print("Норма вектора: ", norm(x))
print("Невязка: ", np.dot(A, x) - l*x)
print()


l, x = spec_bound(A, 1e-3)
print("Противоположная гранца спектра:")
print("Собственное число:", l)
print("Собственный вектор:", x)
print("Норма вектора: ", norm(x))
print("Невязка: ", np.dot(A, x) - l*x)
print()


l, x, k = wielandt_refinement(A, 1e-3)
print("Метод Виланда")
print("Собственное число:", l)
print("Собственный вектор:", x)
print("Количество итераций: ", k)
print("Норма вектора: ", norm(x))
print("Невязка: ", np.dot(A, x) - l*x)