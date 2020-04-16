import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def f(y, t):
    y1, y2 = y
    return [-125 * y1 + 123.45 * y2, 123.45 * y1 - 123 * y2]


y0 = [1, 1]
t = np.linspace(0, 25, 501)
fig = plt.figure(facecolor='white')
# ax=Axes3D(fig)
y1, y2 = odeint(f, y0, t)
plt.plot(y1, y2, linewidth=3)

#  x = np.linspace(x_start, finish, (finish - x_start) / step + 1)
#     y = odeint(f, y0, x)
#     y = np.array(y).flatten()
#     plt.plot(x, y, '-sr', linewidth=4)
#     ax = fig.gca()
#     ax.grid(True)


# def f(y, t):
# y1, y2, y3 = y
# return [y2,0.1*y2-y1*(y3-1)-y1**3,y1*y2-0.1*y3]
# y0=[1,1,0]
# t = np.linspace(0,25,501)
# fig = plt.figure(facecolor='white')
# ax=Axes3D(fig)
# [y1,y2,y3]=odeint(f, y0, t, full_output=False).T
# ax.plot(y1,y2,y3,linewidth=3)


plt.show()
