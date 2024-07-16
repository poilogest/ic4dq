import numpy as np
from scipy.optimize import minimize
import sys
import subprocess
import sympy as sp



N = 5
d = np.full(N, 12000)
w = np.full(N, 500000)
epsilon = np.full(N, 0.2)
w_ = []
epsilon_ = []

g_type = ''
ws = [500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
epsilons = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

ks = [-0.98578142, -0.99297994, -1.02273224, -0.8076]
ms = [0.9801337, 0.91568112, 0.9782473, 0.61111111]
as_ = [-0.60970234, -1.78189984, -2.33489694, -1.91288555]
bs = [-0.13137021, -0.05998946, -0.12081977, 0.33559831]
cs = [0.97933196, 0.88676973, 0.97138007, 0.5905389]
Ns = [5, 5, 10, 10]


k = 0
m = 0
a = 0
b = 0
c = 0

#k = -0.98578142  # 线性时拟合直线的斜率 (kx+m)
#m = 0.9801337  # 线性时拟合直线的截距 (kx+m)
#a = -0.60970234  # 二次时拟合曲线的二次项系数 (ax^2+bx+c)
#b = -0.13137021  # 二次时拟合曲线的一次项系数 (ax^2+bx+c)
#c = 0.97933196  # 二次时拟合曲线的常数项系数 (ax^2+bx+c)

# k = -0.99297994
# m = 0.91568112
# a = -1.78189984
# b = -0.05998946
# c = 0.88676973

# k = -0.88167381
# m = 0.86727314
# a = -1.73815392
# b = 0.15955684
# c = 0.88569711

f_c = -1  # f(x) = f_c * ln(x)


class NE:
    def __init__(self):
        pass
    def f(self, x):
        return f_c * sp.ln(x)
    def best_response(self, d, epsilon, w, f, g, sigma):
        """
        :param d: (d1, d2, ..., dn)
        :param epsilon: (epsilon1, epsilon2, ..., epsilonn)
        :param w: (w1, w2, ..., wn)
        :param f: function f
        :param g: function g
        :param sigma: convergence threshold
        :return: (x1, x2, ..., xn)
        """
        # global x_axis
        # print("epsilon: ", epsilon)
        x = epsilon.copy()
        flag = 0
        while flag == 0:
            flag = 1
            for i in range(len(d)):
                # print("-----------------")
                # print("i: ", i)
                zero = self.zero_point(i, w, d, epsilon, x, f, g)
                # print("zero: ", zero)
                if abs(zero - x[i]) > sigma:
                    flag = 0
                x[i] = zero
                # print("x: ", x)
                # y1.append(x[0])
                # y2.append(x[1])
                # y3.append(x[2])
                # y4.append(x[3])
                # y5.append(x[4])
                # x_axis = x_axis + 1

            # print('=================================')

        return x


    def zero_point(self, i, w, d, epsilon, x, f, g):
        """
        :param d: (d1, d2, ..., dn)
        :param epsilon: (epsilon1, epsilon2, ..., epsilonn)
        :param x: (x1, x2, ..., xn)
        :param f: function f
        :param g: function g
        :return: zero: zero point
        """
        xi = sp.Symbol('xi')  # 定义xi变量
        sum_d = sum([d[j] for j in range(len(d))])  # sum_d = d1 + d2 + ... + dn

        sum_dx = sum([d[j] * x[j] for j in range(len(epsilon)) if j != i])
        sum_dx = sum_dx + d[i] * xi

        g = g(xi, g_type)
        dg = sp.diff(g, xi)  # dg = g'(xi)
        f = self.f(xi)
        df = sp.diff(f, xi)  # df = f'(xi)

        h = ((d[i] * w[i] / sum_d) * dg.subs(xi, sum_dx / sum_d)) - (d[i] * df)

        # print("h: ", h)
        val_i = h.subs(xi, epsilon[i])
        val_0 = h.subs(xi, 0)

        zero = sp.solve(h, xi)
        # print("zero: ", zero)

        if sp.Ge(val_i, 0):
            return epsilon[i]

        if val_0 != sp.zoo and sp.sign(val_0) <= 0:
            return 0

        if len(zero) == 1:
            return zero[0]
        else:
            for z in zero:
                if sp.sign(z) > 0:
                    return z
    def work(self):
        sigma = 0.00001
        return self.best_response(d, epsilon, w, f, g, sigma)



def init(argv):
    global w_, w
    global epsilon_, epsilon
    global g_type, d
    global N, k, m, a, b, c
    
    if(len(argv) != 5):
        print(f"argument error")
        sys.exit("Exiting the program")
    
    N = Ns[int(argv[4])]
    #print(N)
    d = np.full(N, 12000)
    if argv[1] == 'w': # with w as the independent variable
        w = np.full(N, ws[int(argv[2])])
        epsilon = np.array(truncated_normal(0.15, 0.1, N, 0.1, 0.2))
    elif argv[1] == 'epsilon': #with epsilon as the independent variable
        epsilon = np.full(N, epsilons[int(argv[2])])
        w = np.array(truncated_normal(1000000, 10000, N, 500000, 1500000))
    
    g_type = argv[3]
    
    k = ks[int(argv[4])]
    m = ms[int(argv[4])]
    a = as_[int(argv[4])]
    b = bs[int(argv[4])]
    c = cs[int(argv[4])]

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

def truncated_normal(mean, stddev, size, lower, upper):
    raw_values = np.random.normal(mean, stddev, size)
    values = np.clip(raw_values, lower, upper)
    return values


def initialize_parameters(N):
    gamma = np.random.uniform(0, 1, size=(N, N)) * epsilon
    pi = np.random.uniform(0, 1, size=(N, N))
    return gamma, pi


def compute_Vrho(gamma_n, gamma, pi, n, rho):
    gamma_ = gamma
    gamma_[n] = gamma_n
    p, gamma_hat = compute_P(gamma_n, gamma, n)
    sum1 = 0
    for j in range(N):
        sum1 = sum1 + gamma_hat[j] * (pi[(n + 1) % N][j] - pi[(n + 2) % N][j])
    sum2 = 0
    for i in range(N):
        for j in range(N):
            sum2 = sum2 + (gamma_[j - 2][i] - gamma_[j - 1][i]) * (gamma_[j - 2][i] - gamma_[j - 1][i])

    return p[n] + sum1 - rho * sum2


def objective_function(gamma_n, gamma, pi, n, rho):
    return -compute_Vrho(gamma_n, gamma, pi, n, rho)


def compute_P(gamma_n, gamma, n):
    gamma_ = gamma
    gamma_[n] = gamma_n
    gamma_hat = [(np.sum(gamma_[:, i]) / N) for i in range(N)]
    x = [(epsilon[i] - gamma_hat[i]) for i in range(N)]

    sum_d = sum([d[j] for j in range(N)])
    sum_dx = sum([d[j] * x[j] for j in range(N)])
    g1 = g(sum_dx / sum_d, g_type)

    C = []
    for i in range(N):
        C.append(d[i] * (f(x[i]) - f(epsilon[i])))
    P = []
    for i in range(N):
        P.append(g1 * w[i] - C[i])
    return P, gamma_hat


def find_optimal_gamma_n(gamma, pi, n, rho):
    result = minimize(objective_function, x0=[0] * N, args=(gamma, pi, n, rho),
                      bounds=[(0, epsilon[i]) for i in range(N)])
    return result.x if result.success else None


def distributed_algorithm_cross_silo_FL(N, eta=0.01, rho=0.01, phi=1e-5, max_iterations=2000):
    gamma, pi = initialize_parameters(N)
    t = 0
    convg_indicator = 0

    while convg_indicator == 0 and t < max_iterations:
        # Organizations submit their parameters
        gamma_last = gamma.copy()
        pi_last = pi.copy()

        # Central server sends the parameters to each organization
        for n in range(N):
            # For each organization in parallel
            gamma_hat = find_optimal_gamma_n(gamma_last, pi_last, n, rho)

            if gamma_hat is not None:
                gamma[n] = gamma[n] + eta * (gamma_hat - gamma[n])
                pi[n] = pi[n] + rho * eta * (gamma[(n - 2) % N] - gamma[(n - 1) % N])

        t += 1

        if np.max(np.abs(gamma - gamma_last)) <= phi:
            convg_indicator = 1

    return gamma, pi, t


if __name__ == '__main__':
    acc_final = 0
    sw_final = 0
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0
    x5 = 0

    #w_ = []
    #w_tmp = 1100000
    #for i in range(5):
    #    w_.append(w_tmp)

    #epsilon_ = []
    # epsilon_tmp = 0.2
    # for i in range(5):
    #     epsilon_.append(epsilon_tmp)
    ne = NE()
    for i in range(100):
        # w_ = truncated_normal(1000000, 10000, 5, 500000, 1500000)
        #epsilon_ = truncated_normal(0.15, 0.1, 5, 0.1, 0.2)
        init(sys.argv)
        #w = np.array(w_)
        #epsilon = np.array(epsilon_)
        gamma, pi, iterations = distributed_algorithm_cross_silo_FL(N)
        #print(f"Converged gamma: {gamma}")
        #print(f"iteration: {iterations}")
        gamma_hat = [(np.sum(gamma[:, i]) / N) for i in range(N)]
        x = [(epsilon[i] - gamma_hat[i]) for i in range(N)]
        result_array = np.array(ne.work())  # Assuming result is a space-separated string
        # Update x with 50% probability
        for j in range(N):
            if np.random.rand() < 0.5:
                x[j] = result_array[j]
        #print(x)
        x1 = x1 + x[0]
        x2 = x2 + x[1]
        x3 = x3 + x[2]
        x4 = x4 + x[3]
        x5 = x5 + x[4]

        sum_d = sum([d[j] for j in range(N)])
        sum_dx = sum([d[j] * x[j] for j in range(N)])
        acc = g(sum_dx / sum_d, g_type)
        acc_final = acc_final + acc
        #print("acc: " + str(acc))

        C = []
        for i in range(N):
            C.append(d[i] * (f(x[i]) - f(epsilon[i])))
        P = []
        for i in range(N):
            P.append(acc * w[i] - C[i])
        h = np.sum(P)
        sw_final = sw_final + h
        #print("social welfare: " + str(h))
        #print('----------------------------')

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

    #with open("results_acc.txt", "a") as acc_file:
    #    acc_file.write(f'{sys.argv[1]},{sys.argv[2]},{sys.argv[3]},{sys.argv[4]},{str(acc_final / 100)}\n')

    #with open("results_sw.txt", "a") as sw_file:
    #    sw_file.write(f'{sys.argv[1]},{sys.argv[2]},{sys.argv[3]},{sys.argv[4]},{str(sw_final / 100)}\n')
