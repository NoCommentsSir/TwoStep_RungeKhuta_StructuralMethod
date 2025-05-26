import numpy as np
from scipy import integrate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numpy.linalg import solve, norm

def right_y1(x):
    return np.cos(x)

def right_y2(x):
    return -np.sin(x)

def f1(x, y2):
  return y2

def f2(x, y1):
  return -y1

# Функция, запускающая построенный метод 4[2,1]
def TwoStepYMRK21(h):
    # Инициализация коэффициентов метода
    b11 = 16/9
    b12 = 14/45
    a_1121 = 0
    a_1122 = 0
    a_1221 = -1/8
    a_1222 = 5/24
    a1221 = 2/3
    b_11 = -64/45
    c11 = 0
    c12 = 1
    u11 = 1
    u12 = 5/4
    v1 = 4/3

    b21 = 64/105
    a_2111 = -63/20
    a2111 = 231/80
    b_21 = 2/35
    b_22 = 8/15
    c21 = 3/4
    u21 = 161/80
    v2 = 4/5

    # Функция для расчета корректирующих функций на текущем щаге
    def K_j(x, y_1, y_2, y1, y2, h, K_, i):
        # Переворот по условию
        if i%2 == 0: # четный
          K11 = f1(x + c11*h, (1-u11)*y_2 + u11*y2 + a_1121*K_[0]*h + a_1122*K_[2]*h)
          K21 = f2(x + c21*h, (1-u21)*y_1 + u21*y1 + a_2111*K_[1]*h + a2111*K11*h)
          K12 = f1(x + c12*h, (1-u12)*y_2 + u12*y2 + a_1221*K_[0]*h + a_1222*K_[2]*h + a1221*K21*h)
        else: # нечетный
          K11 = f2(x + c11*h, (1-u11)*y_1 + u11*y1 + a_1121*K_[0]*h + a_1122*K_[2]*h)
          K21 = f1(x + c21*h, (1-u21)*y_2 + u21*y2 + a_2111*K_[1]*h + a2111*K11*h)
          K12 = f2(x + c12*h, (1-u12)*y_1 + u12*y1 + a_1221*K_[0]*h + a_1222*K_[2]*h + a1221*K21*h)
        return np.array([K11, K21, K12])


    y_values1 = [1]
    y_values2 = [0]
    x_values = [0]

    # Для запуска метода берем значения аналитического решения на первом и втором шаге
    y_10 = right_y1(0)
    y_20 = right_y2(0)
    y_11 = right_y1(h)
    y_21 = right_y2(h)
    K_011 = f1(0, y_20)
    K_021 = f2(0 + c21*h, right_y1(0 + c21*h))
    K_012 = f1(0 + c12*h, right_y2(0 + c12*h))

    y_values1.append(y_11)
    y_values2.append(y_21)
    x_values.append(h)
    K_ = [K_011, K_021, K_012]
    i = 1

    # Основной цикл программы
    while x_values[i] < 5:
        K = K_j(x_values[i], y_values1[i-1], y_values2[i-1], y_values1[i], y_values2[i], h, K_, i)
        # Нахождение приращений по двум переменным (приращение по y вычисляется с учетом переворота)
        x_i = x_values[i] + h
        if i%2 == 0: # четный
          y_1i = (1-v1)*y_values1[i-1] + y_values1[i]*v1 + h * b_11 * K_[1] + h * b11 * K[0] + h * b12 * K[2]
          y_2i = (1-v2)*y_values2[i-1] + y_values2[i]*v2 + h * b_21 * K_[0] + h * b_22 * K_[2] + h * b21 * K[1]
        else: # нечетный
          y_2i = (1-v1)*y_values2[i-1] + y_values2[i]*v1 + h * b_11 * K_[1] + h * b11 * K[0] + h * b12 * K[2]
          y_1i = (1-v2)*y_values1[i-1] + y_values1[i]*v2 + h * b_21 * K_[0] + h * b_22 * K_[2] + h * b21 * K[1]
        x_values.append(x_i)
        y_values1.append(y_1i)
        y_values2.append(y_2i)
        i += 1
        K_ = K
    return y_values1, y_values2, np.array(x_values)

def TwoStepYMRK32(h):
    b11 = .46201175820562150302
    b12 = .58765851541910606551
    b13 = .12319158850369202261
    a_1121 = 0
    a_1122 = 0
    a_1123 = 0
    a_1221 = .46673551246537396122
    a_1222 = 2.5091153415453527436
    a_1223 = -1.6855578947368421053
    a_1321 = -.3107368830421128927
    a_1322 = -1.548087696504441339
    a_1323 = .600840784803119536
    a1221 = 1.8985238828313785584
    a1321 = 0.662905914242864731e-1
    a1322 = 1/2
    b_11 = -.15029775046263152711
    b_12 = -.35578078845998213137
    c11 = 0
    c12 = 3/5
    c13 = 1
    u11 = 1
    u12 = -1.5888168421052631579
    u13 = 2.691693203319148225
    v1 = 1.3332166767941940674

    b21 = .6280034078968568874
    b22 = .47255616263045985812
    a_2111 = -.35655253824040852900
    a_2112 = -.61415784047474749581
    a_2211 = .3788514177466904890
    a_2212 = .3897300622636323615
    a2111 = .37676909155128063723
    a2211 = .3471733284562585510
    a2212 = .50380105797428403882
    b_21 = -0.3348428475175745004e-1
    b_22 = -.1441940072960781862
    b_23 = -.1033083072392498876
    c21 = 3/16
    c22 = .80689030824726193937
    u21 = 1.7814412871638753876
    u22 = .1873344418063964990
    v2 = 1.1804270287597687786

    def K_j(x, y_1, y_2, y1, y2, h, K_, i):
        if i%2 == 0: # четный
          K11 = f1(x + c11*h, (1-u11)*y_2 + u11*y2 + a_1121*K_[0]*h + a_1122*K_[2]*h + a_1123*K_[4]*h)
          K21 = f2(x + c21*h, (1-u21)*y_1 + u21*y1 + a_2111*K_[1]*h + a_2112*K_[3]*h + a2111*K11*h)
          K12 = f1(x + c12*h, (1-u12)*y_2 + u12*y2 + a_1221*K_[0]*h + a_1222*K_[2]*h + a_1223*K_[4]*h + a1221*K21*h)
          K22 = f2(x + c22*h, (1-u22)*y_1 + u22*y1 + a_2211*K_[1]*h + a_2212*K_[3]*h + a2211*K11*h + a2212*K12*h)
          K13 = f1(x + c13*h, (1-u13)*y_2 + u13*y2 + a_1321*K_[0]*h + a_1322*K_[2]*h + a_1323*K_[4]*h + a1321*K21*h + a1322*K22*h)
        else: # нечетный
          K11 = f2(x + c11*h, (1-u11)*y_1 + u11*y1 + a_1121*K_[0]*h + a_1122*K_[2]*h + a_1123*K_[4]*h)
          K21 = f1(x + c21*h, (1-u21)*y_2 + u21*y2 + a_2111*K_[1]*h + a_2112*K_[3]*h + a2111*K11*h)
          K12 = f2(x + c12*h, (1-u12)*y_1 + u12*y1 + a_1221*K_[0]*h + a_1222*K_[2]*h + a_1223*K_[4]*h + a1221*K21*h)
          K22 = f1(x + c22*h, (1-u22)*y_2 + u22*y2 + a_2211*K_[1]*h + a_2212*K_[3]*h + a2211*K11*h + a2212*K12*h)
          K13 = f2(x + c13*h, (1-u13)*y_1 + u13*y1 + a_1321*K_[0]*h + a_1322*K_[2]*h + a_1323*K_[4]*h + a1321*K21*h + a1322*K22*h)
        return np.array([K11, K21, K12, K22, K13])


    y_values1 = [1]
    y_values2 = [0]
    x_values = [0]

    y_10 = right_y1(0)
    y_20 = right_y2(0)
    y_11 = right_y1(h)
    y_21 = right_y2(h)
    K_011 = f1(0, y_20)
    K_021 = f2(0 + c21*h, right_y1(0 + c21*h))
    K_012 = f1(0 + c12*h, right_y2(0 + c12*h))
    K_022 = f2(0 + c22*h, right_y1(0 + c22*h))
    K_013 = f1(0 + c13*h, right_y2(0 + c13*h))

    y_values1.append(y_11)
    y_values2.append(y_21)
    x_values.append(h)
    K_ = [K_011, K_021, K_012, K_022, K_013]
    i = 1

    while x_values[i] < 5:
        K = K_j(x_values[i], y_values1[i-1], y_values2[i-1], y_values1[i], y_values2[i], h, K_, i)
        x_i = x_values[i] + h
        if i%2 == 0: # четный
          y_1i = (1-v1)*y_values1[i-1] + y_values1[i]*v1 + h * b_11 * K_[1] + h * b_12 * K_[3] + h * b11 * K[0] + h * b12 * K[2] + h * b13 * K[4]
          y_2i = (1-v2)*y_values2[i-1] + y_values2[i]*v2 + h * b_21 * K_[0] + h * b_22 * K_[2] + h * b_23 * K_[4] + h * b21 * K[1] + h * b22 * K[3]
        else: # нечетный
          y_2i = (1-v1)*y_values2[i-1] + y_values2[i]*v1 + h * b_11 * K_[1] + h * b_12 * K_[3] + h * b11 * K[0] + h * b12 * K[2] + h * b13 * K[4]
          y_1i = (1-v2)*y_values1[i-1] + y_values1[i]*v2 + h * b_21 * K_[0] + h * b_22 * K_[2] + h * b_23 * K_[4] + h * b21 * K[1] + h * b22 * K[3]
        x_values.append(x_i)
        y_values1.append(y_1i)
        y_values2.append(y_2i)
        i += 1
        K_ = K
    return y_values1, y_values2, np.array(x_values)

#График аналитического и методического решений метода 4[2,1]
import math
_, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

y_array1, y_array2, x_array = TwoStepYMRK21(1e-3)
axes[0][0].plot(x_array, right_y1(x_array), color='red', label='y1 аналитическое')
axes[0][0].legend()
axes[0][1].plot(x_array, right_y2(x_array), color='red', label='y2 аналитическое')
axes[0][1].legend()
axes[1][0].plot(x_array, y_array1, label='y1 методическое')
axes[1][0].legend()
axes[1][1].plot(x_array, y_array2, label='y2 методическое')
axes[1][1].legend()

#График зависимости глобальной методической погрешности от длины шага и прямая с наклоном равной требуемому порядку точности
from math import *
array_dec1 = []
array_dec2 = []
h_arr = []
for k in range(1, 10):
    h_arr.append(log10(1 / (2 ** k)))
    y_dec1, y_dec2, _ = TwoStepYMRK21(1 / (2 ** k))
    n1 = norm(
        np.array(y_dec1[-1]) - np.array(right_y1(5))
    )
    n2 = norm(
        np.array(y_dec2[-1]) - np.array(right_y2(5))
    )
    array_dec1.append(log10(n1))
    array_dec2.append(log10(n2))
y_line = 4*np.array(h_arr)

_, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axes[0].plot(h_arr, y_line, color = 'r')
axes[0].plot(h_arr, array_dec1, color = 'b')
axes[0].legend(['Прямая с наклоном в 4', 'График погрешности по 1 компоненте'])
axes[1].plot(h_arr, y_line, color = 'r')
axes[1].plot(h_arr, array_dec1, color = 'g')
axes[1].legend(['Прямая с наклоном в 4', 'График погрешности по 2 компоненте'])

# Пример вызова метода 4[2,1]
y_array1, y_array2, x_array = TwoStepYMRK21(1e-3)

# Пример вызова метода 6[3,2]
y_array1, y_array2, x_array = TwoStepYMRK32(1e-3)