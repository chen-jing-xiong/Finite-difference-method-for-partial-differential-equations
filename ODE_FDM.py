"""
常微分方程
u^(2)(x)=-(pi^2)*cos(pi*x)  0<x<1
u(0)=1,u(1)=-1
一般常微分方程
exact_solution=cos(pi*x)
"""
import numpy as np
import matplotlib.pyplot as plt

class ODESolver:
    def __init__(self, n, a, b):
        """
        :param n: 网格点数A
        :param a: 左边界条件 u(0) 的值
        :param b: 右边界条件 u(1) 的值
        """
        self.n = n
        self.h=1/n
        self.a = a
        self.b = b
        self.x = self.generate_grid()
        self.A = self.build_A()
        self.F = self.build_F()
        self.U = None
        self.exact_solution = None
        self.error = None

    def f(self, x):
        """
        定义微分方程的右侧函数
        :param x: 自变量
        :return: 函数值
        """
        return -(np.pi ** 2) * np.cos(np.pi * x)

    def generate_grid(self):
        """
        生成网格点
        :return: 包含网格点的数组
        """
        x = np.zeros(self.n - 1)
        for i in range(self.n - 1):
            x[i] = (i + 1) / self.n
        return x

    def build_A(self):
        """
        构建系数矩阵 A。
        :return: 系数矩阵 A
        """
        main_diag = np.ones(self.n - 1, dtype=float) * (-2) * (1/self.h**2)
        off_diag = np.ones(self.n - 2) * (1/self.h**2)
        return np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    def build_F(self):
        """
        构建线性方程组的右端向量 F。
        :return: 右端向量 F
        """
        F = self.f(self.x)
        F[0] = F[0] - (self.a * (1/self.h**2))
        F[-1] = F[-1] - (self.b * (1/self.h**2))
        return F

    def solve(self):
        """
        求解线性方程组，得到数值解。
        """
        self.U = np.linalg.solve(self.A, self.F)
        self.exact_solution = np.cos(np.pi * self.x)
        self.error = self.exact_solution - self.U

    def plot_results(self):
        """
        绘制数值解、精确解和误差的图像。
        """
        plt.subplot(131)
        plt.plot(self.x, self.U, marker='o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Solution U')
        plt.grid(True)

        plt.subplot(132)
        plt.plot(self.x, self.exact_solution, marker='o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('exact solution ')
        plt.grid(True)

        plt.subplot(133)
        plt.plot(self.x, self.error, marker='o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('error ')
        plt.grid(True)

        plt.show()


# 使用示例
if __name__ == "__main__":
    n = 40
    a = 1
    b = -1
    solver = ODESolver(n, a, b)
    solver.solve()
    solver.plot_results()
    exac_sloution = np.cos(np.pi * solver.x)
    print("误差为：", np.linalg.norm(solver.U - exac_sloution))
