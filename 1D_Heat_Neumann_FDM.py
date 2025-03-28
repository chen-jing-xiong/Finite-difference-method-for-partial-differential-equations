import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, gmres


class PDESolver:
    def __init__(self, n, m, x_left_boundary, x_right_boundary, t_left_boundary, t_right_boundary, g1, g2, u0, beta, f):
        # 初始化空间和时间离散参数
        self.xn = n
        self.xh = (x_right_boundary - x_left_boundary) / n
        self.x_left = x_left_boundary
        self.x_right = x_right_boundary
        self.x = np.linspace(x_left_boundary, x_right_boundary, n + 1)

        self.tn = m
        self.th = (t_right_boundary - t_left_boundary) / m
        self.t_left = t_left_boundary
        self.t_right = t_right_boundary
        self.t = np.linspace(t_left_boundary, t_right_boundary, m + 1)

        self.g1 = g1
        self.g2 = g2
        self.u0 = u0
        self.f = f
        self.beta = beta

        self.A = self.build_A()
        self.U = np.zeros((self.xn + 1, self.tn + 1))
        self.MSE = None

    def build_A(self):
        # 创建隐式差分格式中的系数矩阵
        xh2 = self.xh ** 2
        th = self.th
        N = self.xn + 1
        main_diag = np.full(N, 1 / th + 2 * self.beta / xh2)
        upper_diag = np.full(N - 1, -self.beta / xh2)
        lower_diag = np.full(N - 1, -self.beta / xh2)

        # 处理第二类边界条件
        # 左边界
        upper_diag[0] = -2 * self.beta / xh2
        # 右边界
        lower_diag[-1] = -2 * self.beta / xh2

        A = diags([main_diag, upper_diag, lower_diag], [0, 1, -1], format='csr')
        return A

    def solve(self):
        self.U[:, 0] = self.u0(self.x)
        # 按时间网格迭代
        for k in range(self.tn):
            # 计算右端项，包括源项和上一时刻的值
            F = self.f(self.x, self.t[k + 1]) + self.U[:, k] / self.th

            # 修正边界点以满足第二类边界条件
            F[0] += 2 * self.beta / self.xh * self.g1(self.t[k + 1])
            F[-1] -= 2 * self.beta / self.xh * self.g2(self.t[k + 1])

            # 解线性方程组得到新一层时间的解
            self.U[:, k + 1], _ = gmres(self.A, F)

        # 计算误差
        x, t = np.meshgrid(self.x, self.t, indexing='ij')
        exact_solution = np.cos(np.pi * x) * np.exp( (-np.pi ** 2) * t)
        self.MSE = np.mean((self.U - exact_solution) ** 2)

    def plot_results(self):
        # 生成数值解、精确解和误差的图像
        x, t = np.meshgrid(self.x, self.t, indexing='ij')
        exact_solution = exact_solution = np.cos(np.pi * x) * np.exp( (-np.pi ** 2) * t)
        error = self.U - exact_solution

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
        return np.cos(np.pi * x )


    # 定义源项函数
    def f(x, t):
        return 0


    # 设置空间和时间网格划分数
    n, m = 100, 1000

    # 创建求解器并运行
    solver = PDESolver(n, m, 0, 1, 0, 1, g1, g2, u0, 1, f)
    solver.solve()
    solver.plot_results()
    print(f"均方误差: {solver.MSE}")

