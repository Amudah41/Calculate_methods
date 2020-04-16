import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

x_start = 0
x_finish = 0.5
h = 0.05
y0 = 0
eps = 0.000001

fig = plt.figure(facecolor='white')

def f(x, y):
    return np.cos(2 * x + y) + 1.5 * (x - y)

def y_math(step, finish):
    x = np.linspace(x_start, finish, (finish - x_start) / step + 1)
    y = odeint(f, y0, x)
    y = np.array(y).flatten()
    plt.plot(x, y, '-sr', linewidth=4)
    ax = fig.gca()
    ax.grid(True)
    return y


def Euler_method(step, color):
    x = np.linspace(x_start, x_finish, (x_finish - x_start) / step + 1)
    x = np.array(x).flatten()
    y = [0]*len(x)
    y[0] = y0
    i = 1
    while x[i] < x_finish:
        y[i] = y[i - 1] + step * f(x[i - 1] + step / 2, y[i - 1] + step / 2 * f(x[i - 1], y[i - 1]))
        i += 1
    y[i] = y[i - 1] + step * f(x[i - 1] + step / 2, y[i - 1] + step / 2 * f(x[i - 1], y[i - 1]))
    plt.plot(x, y, color, linewidth=2)
    ax = fig.gca()
    ax.grid(True)
    return  y


def Runge_Kutt(eps,start, finish, step, color):
    x = np.linspace(start, finish, (finish - start) / step + 1)
    x = np.array(x).flatten()
    y = [0]*len(x)
    y[0] = x[0]
    i = 1
    while x[i] < finish:
        k1 = step * f(x[i-1], y[i-1])
        k2 = step * f(x[i-1] + step / 2, y[i-1] + k1 / 2)
        k3 = step * f(x[i-1] + step / 2, y[i-1] + k2 / 2)
        k4 = step * f(x[i-1] + step, y[i-1] + k3)
        y[i] = y[i-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        i += 1
    k1 = step * f(x[i - 1], y[i - 1])
    k2 = step * f(x[i - 1] + step / 2, y[i - 1] + k1 / 2)
    k3 = step * f(x[i - 1] + step / 2, y[i - 1] + k2 / 2)
    k4 = step * f(x[i - 1] + step, y[i - 1] + k3)
    y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    plt.plot(x, y, color, linewidth=2)
    ax = fig.gca()
    ax.grid(True)
    return y

def Runge(s):
    x1 = np.linspace(x_start, x_finish, (x_finish - x_start) / h + 1)
    x1 = np.array(x1).flatten()
    y1 = [0] * len(x1)
    y1[0] = y0
    x2 = np.linspace(x_start, x_finish, (x_finish - x_start) / (h / 2) + 1)
    x2 = np.array(x2).flatten()
    y2 = [0] * len(x2)
    y2[0] = y0
    y1 = Euler_method(h, '-sg')
    y2 = Euler_method(h / 2, '-sb')
    i = 1
    while x1[i] < x_finish:
        R_m = (y2[i * 2] - y1[i]) / (2 ** s - 1)
        y1[i] = y2[i * 2] + R_m
        i += 1
    R_m = (y2[i * 2] - y1[i]) / (2 ** s - 1)
    y1[i] = y2[i * 2] + R_m
    plt.plot(x, y, '-sm', linewidth=2)
    ax = fig.gca()
    ax.grid(True)
    return y1

def Adams_extr (step, finish):
    y = Runge_Kutt(eps, 5 * h, finish - 5 * step,  step, '-sy')
    y_0 = y[-1]
    x_0 = finish - 5 * step
    q_0 = step * f(x_0, y_0)
    y_1 = y_0 + q_0
    x_1 = x_0 + step
    q_1 = step * f(x_1, y_1)
    y_2 = y_1 + (3 * q_1 - q_0) / 2
    x_2 = x_1 + step
    q_2 = step * f(x_2, y_2)
    y_3 = y_2 + (23 * q_2 - 16 * q_1 + 5 * q_0) / 12
    x_3 = x_2 + step
    q_3 = step * f(x_3, y_3)
    y_4 = y_3 + (55 * q_3 - 59 * q_2 + 37 * q_1 - 9 * q_0) / 24
    x_4 = x_3 + step
    q_4 = step * f(x_4, y_4)
    y_5 = y_4 + (1901 * q_4 - 2774 * q_3 + 2616 * q_2 - 1274 * q_1 + 251 * q_0) / 720
    y_main = [0] * int((finish - 5 * step) / step + 1)
    i = 0
    while (y[i] != y[-1]):
        y_main[i] = y[i]
        i+=1
    y_main[i] = y_0
    i += 1
    y_main[i] = y_1
    i += 1
    y_main[i] = y_2
    i += 1
    y_main[i] = y_3
    i += 1
    y_main[i] = y_4
    i += 1
    y_main[i] = y_5
    x = np.linspace(5 * h, 1, 16)
    x = np.array(x).flatten()
    plt.plot(x, y_main, '-sc', linewidth=2)
    ax = fig.gca()
    ax.grid(True)
    return y_main

def Adams_inter (step, finish):
    y = Runge_Kutt(eps, 5 * h, finish - 5 * step,  step, '-sy')
    y_0 = y[-1]
    x_0 = finish - 5 * step
    q_0 = step * f(x_0,y_0)
    y_1 = y_0 + q_0
    x_1 = x_0 + step
    q_1 = step * f(x_1 ,y_1)
    y_2 = y_1 + (q_1 + q_0)/2
    x_2 = x_1 + step
    q_2 = step * f(x_2 ,y_2)
    y_3 = y_2 + (5 * q_2 + 8 * q_1 - q_0) / 12
    x_3 = x_2 + step
    q_3 = step * f(x_3 ,y_3)
    y_4 = y_3 + (9 * q_3 + 19 * q_2 - 5 * q_1 + q_0) / 24
    x_4 = x_3 + step
    q_4 = step * f(x_4 ,y_4)
    y_5 = y_4 + (251 * q_4 + 646 * q_3 - 264 * q_2  + 106 * q_1 - 19 * q_0) / 720
    y_main = [0] * int((finish - 5 * step) / step + 1)
    i = 0
    while (y[i] != y[-1]):
        y_main[i] = y[i]
        i+=1
    y_main[i] = y_0
    i += 1
    y_main[i] = y_1
    i += 1
    y_main[i] = y_2
    i += 1
    y_main[i] = y_3
    i += 1
    y_main[i] = y_4
    i += 1
    y_main[i] = y_5
    x = np.linspace(5 * h, 1, 16)
    x = np.array(x).flatten()
    plt.plot(x, y_main, '-sk', linewidth=2)
    ax = fig.gca()
    ax.grid(True)
    return y_main



y_math1 = y_math(h, 0.5)
print("Точное решение, полученное с помощью встроенного пакета:")
print("y_math : ", y_math1)
print()

y = Euler_method(h / 2, '-sb')
print("Решение, полученное с помощью метода Эйлера с шагом h/2:")
print("y : ", y)
print()

y = Euler_method(h, '-sg')
print("Решение, полученное с помощью метода Эйлера с шагом h:")
print("y : ", y)
print()

y_rev = Runge(4)
print("Решение, уточнённое с помощью метода Рунге:")
print("y_rev : ", y_rev)
print("Невязка:", y_rev - y_math1)
print()

y_RK = Runge_Kutt(eps, 0, 0.5, h, '-sm')
print("Решение, полученное с помощью метода Рунге-Кутта с шагом h:")
print("y_RK : ", y_RK)
print("Невязка:", y_math1 - y_RK)
print()

y_math2 = y_math(h, 1)      #получение массива с точным решением на промежутке [5h,1]
i = 5
y_math21 = [0]*16
while i < 20:
    y_math21[i-5] = y_math2[i]
    i += 1
y_math21 = np.array(y_math21)

y_Ad_ex = Adams_extr(h, 1)
print("Решение, полученное с помощью метода экстрополяционным Адамса с шагом h:")
print("y_Ad_ex : ", y_Ad_ex)
print("Невязка:", y_math21 - y_Ad_ex)
print()

y_Ad_in = Adams_inter(h, 1)
print("Решение, полученное с помощью метода экстрополяционным Адамса с шагом h:")
print("y_Ad_ex : ", y_Ad_in)
print("Невязка:", y_math21 - y_Ad_in)

plt.show()

# 'b' -
# 'g' - зеленый цвет
# красный цвет - y_math
# синий цвет - решение методом Эйлера с шагом h/2
# зеленый цвет - решение методом Эйлера с шагом h
# 'c' - голубой цвет
# 'm' - пурпурный цвет
# 'y' - желтый цвет
# 'k' - черный цвет
# 'w' - белый цвет