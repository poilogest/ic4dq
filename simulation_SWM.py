from scipy.optimize import minimize_scalar
import numpy as np
import random
import math

import sys

n = 5
d = np.full(n, 12000)
w = np.full(n, 500000)
epsilon = np.full(n, 0.2)
w_ = []
epsilon_ = []

g_type = 'q'

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
a = -1.78189984
b = -0.05998946
c = 0.88676973

f_c = -1  # f(x) = f_c * ln(x)

x1 = 0
x2 = 0
x3 = 0
x4 = 0
x5 = 0
acc_final = 0
sw_final =0

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


def H(t, d, epsilon, w, f, g):
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


if __name__ == '__main__':
    #file = open('./w_SWM_linear_5e5.txt', 'w')
    #d = np.array([12000, 12000, 12000, 12000, 12000])
    #w = []
    #w_tmp = 1500000
    #for i in range(5):
    #    w.append(w_tmp)

    #epsilon = []
    # epsilon_tmp = 0.08
    # for i in range(5):
    #     epsilon.append(epsilon_tmp)

    #n = len(d)
    for i in range(100):
        init(sys.argv)
        # w = truncated_normal(1000000, 10000, 5, 500000, 1500000)
        #epsilon = truncated_normal(0.15, 0.1, 5, 0.1, 0.2)
        epsilon_max = max(epsilon)
        x = []
        # w = []
        # w_tmp = 1500000
        # for i in range(n):
        #     w.append(w_tmp)

        #print('epsilon: ' + str(epsilon))
        #file.write('epsilon: ' + str(epsilon) + '\n')

        sum_d = sum([d[j] for j in range(n)])
        sum_w = sum([w[j] for j in range(n)])

        H_arr = []

        res = minimize_scalar(H, args=(d, epsilon, w, f, g), method='bounded', bounds=(0, epsilon_max))
        # print("t0: " + str(res.x))
        # print("H: " + str(-res.fun))
        H_arr.append(res.fun)

        n = len(d)
        for i in range(n):
            x.append(min(epsilon[i], res.x))
        #print("x: " + str(x))
        #file.write("x: " + str(x) + '\n')
        #print(x)
        #print(w)
        #print(epsilon)
        x1 = x1 + x[0]
        x2 = x2 + x[1]
        x3 = x3 + x[2]
        x4 = x4 + x[3]
        x5 = x5 + x[4]

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

        P = 0
        for i in range(n):
            p_i = acc * w[i] - C[i]
            # p_i = acc * w[i]
            P = P + p_i
        #print('P: ' + str(P))
        #file.write('P: ' + str(P) + '\n')
        sw_final = sw_final + P
        #print('----------------------------')
        #file.write('----------------------------' + '\n')


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



    # # print('sqrt理论值: ' + str((f_c * sum_d) / (sum_w * k)))  # linear理论值
    # for i in range(n):
    #     # sqrt理论值
    #     A = 2 * sum_w * a * d[i]
    #     sum_dx = sum([d[j] * x[j] for j in range(len(d)) if j != i])
    #     B = 2 * sum_w * a * sum_dx + sum_w * b * sum_d
    #     C = -(f_c * sum_d * sum_d)
    #     tmp_1 = B * B - 4 * A * C
    #     if tmp_1 < 0:
    #         print("error: B * B - 4 * A * C < 0")
    #     else:
    #         print('sqrt理论值: ' + str((-B - math.sqrt(B * B - 4 * A * C)) / (2 * A)))
