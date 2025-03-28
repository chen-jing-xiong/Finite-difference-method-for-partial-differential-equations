"""
常微分方程
u^(2)(x)+b(x)U^(1)(X)+C(X)U(X)=f(x)  0<x<1

边界条件
aU^(1)(0)+bU(0)=u1(0)
u(1)=u2(1)

下面给出一组参数
u^(2)(x)=-(pi^2)*cos(pi*x)  0<x<1

u^(1)(0)=0,u(1)=-1
精确解
exact_solution=cos(pi*x)
"""
import matplotlib.pyplot as plt
import numpy as np


class ODESolver:
    def __init__(self, n, left_boundary, right_boundary, u1, u2, b_x, c_x, f_x, a, b):
        """

        :param n: 网格点数
        :param left_boundary:左边界
        :param right_bounda: 右边界
        :param u1: 左边界值函数
        :param u2: 右边界值函数
        :param b_x:U^(1)(x)的系数函数
        :param c_x:U(x)的系数函数
        :param f_x:右端函数项
        :param a:边界条件U^(1)系数
        :param b:边界条件U系数
        """
        self.n = n
        self.h = (right_boundary - left_boundary) / n
        self.left = left_boundary
        self.right = right_boundary
        self.x = self.generate_grid()
        self.u1 = u1
        self.u2 = u2
        self.b_x = b_x
        self.c_x = c_x
        self.f_x = f_x
        self.a = a
        self.b = b
        self.A = self.build_A()
        self.F = self.build_F()
        self.U = None

    def generate_grid(self):
        """
        生成网格点
        :return: 包含网格点的数组
        """
        x = np.zeros(self.n)
        for i in range(self.n):
            x[i] = self.left + i * self.h
        return x

    def build_A(self):
        """
        构建系数矩阵 A
        :return: 系数矩阵 A
        """
        c = self.c_x(self.x)
        b1 = self.b_x(self.x[:-1])
        b2 = self.b_x(self.x[1:])

        main_diag = c - np.ones(self.n) * (2 / (self.h ** 2))
        main_diag[0] = main_diag[0] + 2 * self.h * self.b * (
                    (1 / self.h ** 2) - self.b_x(self.x[0]) / 2 * self.h) / self.a

        up_diag = (np.ones(self.n - 1) / (self.h ** 2)) + b1 / (2 * (self.h ** 2))
        up_diag[0] = up_diag[0] + ((1 / self.h ** 2) - self.b_x(self.x[0]) / 2 * self.h)

        down_diag = (np.ones(self.n - 1) / (self.h ** 2)) - b2 / (2 * (self.h ** 2))

        return np.diag(main_diag) + np.diag(up_diag, k=1) + np.diag(down_diag, k=-1)

    def build_F(self):
        """
        构建线性方程组的右端向量 F。
        :return: 右端向量 F
        """
        F = self.f_x(self.x)
        U_minu_1 = 2 * self.h * self.b * ((1 / self.h ** 2) - self.b_x(self.x[0]) / 2 * self.h) / self.a
        F[0] = F[0] + U_minu_1
        F[-1] = F[-1] - self.u2(self.right) * (1 / self.h ** 2 + self.b_x(self.x[-1]) / 2 * self.h)
        return F

    def solve(self):
        """
        求解线性方程组，得到数值解。
        """
        self.U = np.linalg.solve(self.A, self.F)

    def plot_results(self):
        """
        绘制数值解、精确解和误差的图像。
        """

        plt.plot(self.x[1:], self.U[1:], marker='o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Solution U')
        plt.grid(True)

        plt.show()


# 使用示例
if __name__ == "__main__":
    """

    """
    n = 100
    left_boundary = 0
    right_boundary = 1


    def u1(x):
        return 0

    def u2(x):
        return -1

    def b_x(x):
        return 0

    def c_x(x):
        return 0

    def f_x(x):
        return -(np.pi ** 2) * np.cos(np.pi * x)


    a = 1
    b = 0
    solver = ODESolver(n, left_boundary, right_boundary, u1, u2, b_x, c_x, f_x, a, b)
    solver.solve()
    solver.plot_results()
    exac_sloution = np.cos(np.pi * solver.x)
    print("误差为：", np.mean((solver.U[1:] - exac_sloution[1:])**2))
