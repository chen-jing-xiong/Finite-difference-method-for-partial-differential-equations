"""
抛物线方程求解

方程形式：
    u_{t} =\beta*u_{xx}+f(x,t),  a < x < b, t>0

边界条件：
    u(a,t)=g1(t) ,u(b,t)=g2(t)
    u(x,0)=u0(x)

具体问题：
    u_t = u_XX+exp(-t) * sin(2 * pi * x)
    定义域：0 < x < 1, 0<t<1

边界条件：
    u(0,t)=0 ,u(1,t)=0
    u(x,0)=sin(pi*x)

精确解：
    u(x, y) = exp(-pi^ 2 * t) * sin(npi * x) + (exp(-t) * sin(2 * pi * x)) / (4 * pi^2 - 1)

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class PDESolver:
    def __init__(self, n, m, x_left_boundary, x_right_boundary, t_left_boundary, t_right_boundary, g1, g2, u0, beta, f):
        # 初始化空间和时间离散参数
        self.xn = n                                                     # 划分网格数
        self.xh = (x_right_boundary - x_left_boundary) / n              # 网格单元长度
        self.x_left = x_left_boundary                                   # 左边界
        self.x_right = x_right_boundary                                 # 右边界
        self.x = np.linspace(x_left_boundary, x_right_boundary, n + 1)  # x坐标划分点向量

        self.tn = m
        self.th = (t_right_boundary - t_left_boundary) / m
        self.t_left = t_left_boundary
        self.t_right = t_right_boundary
        self.t = np.linspace(t_left_boundary, t_right_boundary, m + 1)


        self.g1 = g1    # 左边界函数
        self.g2 = g2    # 右边界函数
        self.u0 = u0    # 初始条件函数
        self.f = f      # 源项
        self.beta = beta# 扩散系数

        self.A = self.build_A()                       # 构建系数矩阵A
        self.U = np.zeros((self.xn + 1, self.tn + 1)) # 初始化解矩阵
        self.MSE = None                               # 初始化误差

    def build_A(self):
        # 创建隐式差分格式中的系数矩阵
        a = 1 / self.th + 2 * self.beta / self.xh ** 2
        b = -self.beta / self.xh ** 2
        N = self.xn - 1
        # 构建三对角稀疏矩阵
        diagonals = [
            b * np.ones(N - 1),
            a * np.ones(N),
            b * np.ones(N - 1)
        ]
        return diags(diagonals, offsets=[-1, 0, 1]).tocsr()

    def solve(self):
        # 设置初始条件（内部点）
        self.U[1:-1, 0] = self.u0(self.x[1:-1])
        # 设置边界条件
        # self.U[0, :] = self.g1(self.t)
        # self.U[-1, :] = self.g2(self.t)

        # 安装时间网格迭代
        for k in range(self.tn):
            # 计算右端项，包括源项和上一时刻的值
            F = self.f(self.x[1:self.xn], self.t[k + 1]) + self.U[1:self.xn, k] / self.th
            # 修正边界点
            F[0] += (self.beta / self.xh ** 2) * self.g1(self.t[k + 1])
            F[-1] += (self.beta / self.xh ** 2) * self.g2(self.t[k + 1])
            # 解线性方程组得到新一层时间的解
            self.U[1:self.xn, k + 1] = spsolve(self.A, F)

        # 计算误差
        x, t = np.meshgrid(self.x, self.t, indexing='ij')
        exact_solution = np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x) + (np.exp(-t) * np.sin(2 * np.pi * x)) / (4 * np.pi**2 - 1)
        self.MSE = np.mean((self.U[1:-1, :] - exact_solution[1:-1, :]) ** 2)

    def plot_results(self):
        # 生成数值解、精确解和误差的图像
        x, t = np.meshgrid(self.x, self.t, indexing='ij')
        exact_solution = np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x) + (np.exp(-t) * np.sin(2 * np.pi * x)) / (4 * np.pi**2 - 1)
        error = np.abs(self.U - exact_solution)

        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # 绘制数值解
        cp1 = ax[0].contourf(x, t, self.U, 20, cmap='viridis')
        fig.colorbar(cp1, ax=ax[0])
        ax[0].set_title('Numerical Solution')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('t')

        # 绘制精确解
        cp2 = ax[1].contourf(x, t, exact_solution, 20, cmap='viridis')
        fig.colorbar(cp2, ax=ax[1])
        ax[1].set_title('Exact Solution')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('t')

        # 绘制误差
        cp3 = ax[2].contourf(x, t, error, 20, cmap='viridis')
        fig.colorbar(cp3, ax=ax[2])
        ax[2].set_title('Error')
        ax[2].set_xlabel('x')
        ax[2].set_ylabel('t')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 定义边界条件函数
    def g1(t):
        return 0

    def g2(t):
        return 0

    # 定义初始条件函数
    def u0(x):
        return np.sin(np.pi * x)

    # 定义源项函数
    def f(x, t):
        return np.exp(-t) * np.sin(2 * np.pi * x)

    # 设置空间和时间网格划分数
    n, m = 10, 100

    # 创建求解器并运行
    solver = PDESolver(n, m, 0, 1, 0, 1, g1, g2, u0, 1, f)
    solver.solve()
    solver.plot_results()
    print(f"均方误差: {solver.MSE}")

