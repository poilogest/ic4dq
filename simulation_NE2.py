import random
import numpy as np
import sympy as sp
from scipy import integrate
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import math

import sys

n = 10
d = np.full(n, 12000)
w = np.full(n, 500000)
epsilon = np.full(n, 0.2)

g_type = ''

ws = [500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
epsilons = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
ks = [-0.98578142, -0.99297994, -1.02273224, -0.8076]
ms = [0.9801337, 0.91568112, 0.9782473, 0.61111111]
as_ = [-0.60970234, -1.78189984, -2.33489694, -1.91288555]
bs = [-0.13137021, -0.05998946, -0.12081977, 0.33559831]
cs = [0.97933196, 0.88676973, 0.97138007, 0.5905389]
ns = [5, 5, 10, 10]


k = 0
m = 0
a = 0
b = 0
c = 0

f_c = -1  # f(x) = f_c * ln(x)

x1 = 0
x2 = 0
x3 = 0
x4 = 0
x5 = 0
acc_final = 0
sw_final =0
M = 0

def init(argv):
    global w
    global epsilon
    global g_type, d
    global n, k, m, a, b, c
    
    if(len(argv) != 5):
        print(f"argument error")
        sys.exit("Exiting the program")
    
    n = ns[int(argv[4])]
    d = np.full(n, 12000)
    if argv[1] == 'w': # with w as the independent variable
        w = np.full(n, ws[int(argv[2])])
        epsilon = np.array(truncated_normal(0.15, 0.1, n, 0.1, 0.2))
    elif argv[1] == 'epsilon': #with epsilon as the independent variable
        epsilon = np.full(n, epsilons[int(argv[2])])
        w = np.array(truncated_normal(1000000, 10000, n, 500000, 1500000))
    
    g_type = argv[3]
    
    k = ks[int(argv[4])]
    m = ms[int(argv[4])]
    a = as_[int(argv[4])]
    b = bs[int(argv[4])]
    c = cs[int(argv[4])]

def truncated_normal(mean, stddev, size, lower, upper):
    raw_values = np.random.normal(mean, stddev, size)
    values = np.clip(raw_values, lower, upper)
    return values


def f(x):
    x_safe = np.maximum(x, np.finfo(float).eps)
    return f_c * np.log(x_safe)
    return f_c * np.log(x)



def g(x, function_type):
    if function_type == 'l':
        return k * x + m
    elif function_type == 'q':
        return a * (x ** 2) + b * x + c
    else:
        raise ValueError(f"Unknown function type: {function_type}")
def H(t, d, epsilon, w):
    """
       :param t: variable t
       :param d: (d1, d2, ..., dn)
       :param epsilon: (epsilon1, epsilon2, ..., epsilonn)
       :param w: (w1, w2, ..., wn)
       :param f: function f
       :param g: function g
       :return: (x1, x2, ..., xn)
    """
    # print("d: " + str(d))
    n = len(d)
    x = []
    for i in range(n):
        x.append(min(epsilon[i], t))
    # print("x: " + str(x))

    sum_d = sum([d[j] for j in range(n)])
    sum_dx = sum([d[j] * x[j] for j in range(n)])
    g1 = g(sum_dx / sum_d, g_type)

    C = []
    for i in range(n):
        C.append(d[i] * (f(x[i]) - f(epsilon[i])))

    h = 0
    for i in range(n):
        p_i = g1 * w[i] - C[i]
        h = h + p_i

    return -h


def best_response_2(d, epsilon, w, sigma, x_swm):
    x = epsilon.copy()
    flag = 0
    cnt = 0
    while flag == 0:
        flag = 1
        for i in range(len(d)):
            # print("-----------------")
            # print("i: ", i)
            zero = maximize(i, d, epsilon, w, x, x_swm)
            # print("zero: ", zero)
            if abs(zero - x[i]) > sigma:
                cnt = cnt + 1
                # print("abs(zero - x[i]): ", abs(zero - x[i]))
                flag = 0
            x[i] = zero

    return x, cnt


def integrand(x, u, v):
    return (1 / (math.sqrt(v) * math.sqrt(2 * math.pi))) * math.exp(-((x - u) ** 2) / (2 * v))


def objective_function(xi, i, d, epsilon, w, x, x_swm):
    # 定义 P(x) 的表达式
    sum_d = sum([d[j] for j in range(len(d))])  # sum_d = d1 + d2 + ... + dn
    sum_dx = sum([d[j] * x[j] for j in range(len(epsilon)) if j != i])
    sum_dx = sum_dx + d[i] * xi
    avg_x = sum_dx / sum_d

    # print("xi: ", xi)
    # print("avg_x: ", avg_x)
    # print("integrate: ", integrate.quad(lambda y: integrand(y, x_swm[i], 1), x_swm[i] - 0.05 , x_swm[i] + 0.05)[0])
    # print("f(xi): ", f(xi))
    # print("f(epsilon[i]): ", f(epsilon[i]))
    # print("g(avg_x): ", g(avg_x))
    # m = 50000000 # w为横坐标时
    #m = 50000000
    m = 10 * w[i]
    P_x = w[i] * g(avg_x, g_type) - d[i] * (f(xi) - f(epsilon[i])) + m * (integrate.quad(lambda y: integrand(y, xi, 1), x_swm[i] - 0.05 , x_swm[i] + 0.05)[0])
    # print(integrate.quad(lambda y: integrand(y, xi, 1), x_swm[i] - 0.05 , x_swm[i] + 0.05)[0])
    # 返回 P(x) 的相反数，因为我们要最大化 P(x) 就等于最小化 -P(x)
    return -P_x


def maximize(i, d, epsilon, w, x, x_swm):
    initial_guess = np.array([0.0])
    # result = minimize(objective_function, initial_guess, args=(i, d, epsilon, w, x, x_swm), bounds=[(0, epsilon[i])])
    result = minimize_scalar(objective_function, args=(i, d, epsilon, w, x, x_swm), method='bounded', bounds=(0, epsilon[i]))
    # print("result: ", result)
    # print("result.x[0]: ", result.x[0])
    return result.x


if __name__ == '__main__':
    # 写文件
    #file = open('w_NE2_quadratic_5e5.txt', 'w')
    #d = [12000, 12000, 12000, 12000, 12000]
    #w = []
    #w_tmp = 1500000
    #for i in range(5):
        #w.append(w_tmp)
    #epsilon = []
    # epsilon_tmp = 0.2
    # for i in range(5):
    #     epsilon.append(epsilon_tmp)

    sigma = 0.00001
    converge_iter = []

    #print((integrate.quad(lambda y: integrand(y, -1, 1), 0 - 0.05 , 0 + 0.05)[0]))

    
    for i in range(100):
        # w = truncated_normal(1000000, 10000, 5, 500000, 1500000)
        init(sys.argv)
        #epsilon = truncated_normal(0.15, 0.1, 5, 0.1, 0.2)

                     # w = []
                     # w_tmp = 1500000
                    # for i in range(5):
                    #     w.append(w_tmp)

        #print('epsilon: ' + str(epsilon))
        #file.write('epsilon: ' + str(epsilon) + '\n')

        x_swm = []
        epsilon_max = max(epsilon)
        res = minimize_scalar(H, args=(d, epsilon, w), method='bounded', bounds=(0, epsilon_max))
        n = len(d)
        for i in range(n):
            x_swm.append(min(epsilon[i], res.x))
        
        x, cnt = best_response_2(d, epsilon, w, sigma, x_swm)
        print(x_swm)
        print(x)
        converge_iter.append(cnt)
        # x = [0, 0, 0, 0, 0]
        #print('x: ' + str(x))
        #file.write('x: ' + str(x) + '\n')
        x1 = x1 + x[0]
        x2 = x2 + x[1]
        x3 = x3 + x[2]
        x4 = x4 + x[3]
        x5 = x5 + x[4]

        n = len(x)
        sum_dx = sum([d[j] * x[j] for j in range(n)])
        sum_d = sum([d[j] for j in range(n)])
        x_ = sum_dx / sum_d
        acc = g(x_, g_type)
        #print('Acc: ' + str(acc))
        #file.write('Acc: ' + str(acc) + '\n')
        acc_final = acc_final + acc

        C = []
        for i in range(n):
            C.append(d[i] * (f(x[i]) - f(epsilon[i])))
            # C.append(0)

        #print('C: ' + str(C))

        P = 0
        for i in range(n):
            p_i = acc * w[i] - C[i]
            # p_i = acc * w[i]
            P = P + p_i
        #print('P: ' + str(P))
        sw_final = sw_final + P
        #file.write('P: ' + str(P) + '\n')
        #print('--------------')
        #file.write('--------------' + '\n')

    x_final = [x1 / 100, x2 / 100, x3 / 100, x4 / 100, x5 / 100]
    #print('x1 avg: ' + str(x1 / 100))
    #print('x2 avg: ' + str(x2 / 100))
    #print('x3 avg: ' + str(x3 / 100))
    #print('x4 avg: ' + str(x4 / 100))
    #print('x5 avg: ' + str(x5 / 100))
    #print('final avg: ' + str((x1 + x2 + x3 + x4 + x5) / 500))
    #print('final max avg: ' + str(max(x1, x2, x3, x4, x5) / 100))
    #print('final min avg: ' + str(min(x1, x2, x3, x4, x5) / 100))
    print('Acc avg: ' + str(acc_final / 100))
    print('SW avg: ' + str(sw_final / 100))
    #print('converge: ' + str(converge_iter) )

    #file.write('x1 avg: ' + str(x1 / 100) + '\n')
    #file.write('x2 avg: ' + str(x2 / 100) + '\n')
    #file.write('x3 avg: ' + str(x3 / 100) + '\n')
    #file.write('x4 avg: ' + str(x4 / 100) + '\n')
    #file.write('x5 avg: ' + str(x5 / 100) + '\n')
    #file.write('final avg: ' + str((x1 + x2 + x3 + x4 + x5) / 500) + '\n')
    #file.write('final max avg: ' + str(max(x1, x2, x3, x4, x5) / 100) + '\n')
    #file.write('final min avg: ' + str(min(x1, x2, x3, x4, x5) / 100) + '\n')
    #file.write('Acc avg: ' + str(acc_final / 100) + '\n')
    #file.write('SW avg: ' + str(sw_final / 100) + '\n')


    # expect_x = []  # 根据论文里的公式计算出来的x1,..,xn的理论值
    # for i in range(len(x)):
    #     sum_d = sum([d[j] for j in range(len(d))])
    #     # # linear理论值
    #     # expect_x.append((f_c * sum_d) / (w[i] * k))
    #
    #     # sqrt理论值
    #     A = 2 * w[i] * a * d[i]
    #     sum_dx = sum([d[j] * x[j] for j in range(len(d)) if j != i])
    #     B = 2 * w[i] * a * sum_dx + w[i] * b * sum_d
    #     C = -(f_c * sum_d * sum_d)
    #     tmp_1 = B * B - 4 * A * C
    #     if tmp_1 < 0:
    #         print("error: B * B - 4 * A * C < 0")
    #     else:
    #         expect_x.append((-B - math.sqrt(B * B - 4 * A * C)) / (2 * A))
    #
    # print('expect_x: ' + str(expect_x))
    #
    # x_axis_ = []
    # for t in range(x_axis):
    #     x_axis_.append(t)
    # print('x = '+ str(x_axis_))
    # print('y1 = ' + str(y1))
    # print('y2 = ' + str(y2))
    # print('y3 = ' + str(y3))
    # print('y4 = ' + str(y4))
    # print('y5 = ' + str(y5))
