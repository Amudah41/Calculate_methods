from numpy.linalg import eig
import copy
from functools import partial
import numpy as np
from numpy.linalg import det
from numpy.linalg import norm, solve



A = np.array([[9.016024, 1.082197, -2.783575], [1.082197, 6.846595, 0.647647], [-2.783575, 0.647647, 5.432541]])
b = np.array([-1.340873, 4.179164, 5.478007])

norm_inf = partial(norm, ord=np.inf)

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

def transfer_system(A, b):
    assert A.shape[0] == A.shape[1] == len(b)
    H = np.zeros_like(A)
    g = np.zeros_like(b)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i, j] = 0
            else:
                H[i, j] = -A[i, j] / A[i, i]
        g[i] = b[i] / A[i, i]

    return H, g


def simple_iteration(H, g, k, x0=None):
    assert H.shape[0] == H.shape[1] == len(g)
    if x0 is None:
        x0 = np.zeros_like(g)
    x_cur = x0
    for i in range(k):
        x_cur = np.dot(H, x_cur) + g
    return x_cur


def apost_est(x_k, x_j, H):
    return norm_inf(H) * norm_inf(x_k - x_j) / (1 - norm_inf(H))

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


def seidel(H, g, k, x0=None):
    assert H.shape[0] == H.shape[1] == len(g)
    if x0 is None:
        x0 = np.zeros_like(g)
    n = H.shape[0]
    Hl, Hr = np.zeros_like(H), np.zeros_like(H)
    for i in range(n):
        Hr[i, i:] = H[i, i:]
        Hl[i, :i] = H[i, :i]
    E = np.eye(n)
    He = invert(E - Hl)

    x_cur = x0
    for i in range(k):
        x_cur = np.dot(He, np.dot(Hr, x_cur)) + np.dot(He, g)

    return x_cur, np.dot(He, Hr)


def spectre_rad(A):
    eig_numbers = eig(A)[0]
    return np.max(np.abs(eig_numbers))


def lusternik_correction(x_k, x_j, H):
    return x_j + (x_k - x_j) / (1 - spectre_rad(H))


def relaxation(H, g, k, x0=None):
    assert H.shape[0] == H.shape[1] == len(g)
    if x0 is None:
        x0 = np.zeros_like(g)
    n = H.shape[0]
    sr = spectre_rad(H)
    q = 2.0 / (1 + np.sqrt(1 - sr**2))
    x_cur = x0
    for l in range(k):
        x_next = np.zeros_like(x_cur)
        for i in range(n):
            s1 = 0
            for j in range(i):
                s1 += (H[i][j] * x_next[j])
            s2 = 0
            for j in range(i, n):
                s2 += (H[i][j] * x_cur[j])
            x_next[i] = x_cur[i] + q * (g[i] - x_cur[i] + s1 + s2)
        x_cur = x_next

    return x_cur



x_star = solve_gauss_enhanced(A, b)
print("Точное решение: ", x_star);
print();
H, g = transfer_system(A, b)
print("Норма матрицы H: ", norm_inf(H))
print();

x_10 = simple_iteration(H, g, 10)
print("Решение методом простой итерации", x_10)
print();

x_9 = simple_iteration(H, g, 9)
print("Апостериорная оценка для метода простой итерации: ", apost_est(x_10, x_9, H))
print("Фактическая погрешность для метода простой итерации: ", norm_inf(x_10 - x_star))
print();

x_10_seid, H_seid = seidel(H, g, 10)
print("Приближение методом Зейделя:", x_10_seid);
print("Разница приближения методом Зейделя и метода простой итерации:", norm_inf(x_10_seid-x_10));
print("Фактическая погрешность для метода Зейделя: ", norm_inf(x_10_seid - x_star))
print();

print("Спектральный радиус матрицы перехода: ", spectre_rad(H_seid))
x_9_seid = seidel(H, g, 9)[0]
x_l = lusternik_correction(x_10, x_9, H)
print("Уточнение Люстерника для приближения по методу простой итерации: ", norm_inf(x_l - x_star))
x_l = lusternik_correction(x_10_seid, x_9_seid, H)
print("Уточнение Люстерника для приближения по методу Зейделя: ", norm_inf(x_l - x_star))
print();

x_10_rel = relaxation(H, g, 10)
print("Решение методом верхней релаксации:", x_10_rel)
print("Фактическая погрешность для метода верхней релаксации: ", norm_inf(x_10_rel - x_star))
