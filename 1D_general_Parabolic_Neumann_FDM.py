"""
抛物线方程求解

方程形式：
    u_{t} -u_{xx}+b(x,t)u_{x}+c(x,t)u(x,t)=f(x,t),  a < x < b, t>0

边界条件：
    u_x(a,t)=g1(t) ,u_x(b,t)=g2(t)
    u(x,0)=u0(x)

具体问题：
    u_{t} -u_{xx}+2x*u_{x}-u(x,t)=exp^t * (4 * x^2 - 2),  0 < x < 1, t>0

边界条件：
    u_x(0,t)=0 ,u_x(1,t)=2e^t
    u(x,0)=x^2

精确解：
    u(x, y) = (x^2)*e^t

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, gmres


class PDESolver:
    def __init__(self, n, m, x_left_boundary, x_right_boundary, t_left_boundary, t_right_boundary, g1, g2, u0, b, c, f):
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

        self.g1 = g1     # 左边界函数
        self.g2 = g2     # 右边界函数
        self.u0 = u0     # 初始条件函数
        self.b = b       # u_x系数
        self.c = c       # u 系数
        self.f = f       # 源项

        self.U = np.zeros((self.xn + 1, self.tn + 1))
        self.MSE = None

    def build_A(self, k):
        # 创建隐式差分格式中的系数矩阵
        xh2 = self.xh ** 2
        xh = self.xh
        th = self.th
        N = self.xn + 1

        # 计算系数矩阵的对角线元素
        main_diag = np.ones(N) * (1 / th +2 / xh2) + self.c(self.x, self.t[k + 1])
        upper_diag = np.ones(N - 1) * (-1 / xh2) + self.b(self.x[:-1], self.t[k + 1]) / (2 * xh)
        lower_diag = np.ones(N - 1) * (-1 / xh2) - self.b(self.x[1:], self.t[k + 1]) / (2 * xh)

        # 修正边界条件
        upper_diag[0] = -2/xh2
        lower_diag[-1]= -2/xh2

        A = diags([main_diag, upper_diag, lower_diag], [0, 1, -1], format='csr')
        return A

    def solve(self):
        self.U[:, 0] = self.u0(self.x)
        for k in range(self.tn):
            F = self.f(self.x, self.t[k + 1]) + self.U[:, k] / self.th

            # 边界条件处理 (Neumann条件)
            F[0] += -2 / self.xh * self.g1(self.t[k + 1])-self.b(self.x[0], self.t[k + 1])*self.g1(self.t[k + 1])
            F[-1] -= -2/ self.xh * self.g2(self.t[k + 1])+self.b(self.x[-1], self.t[k + 1])*self.g2(self.t[k + 1])

            A = self.build_A(k)

            # 选择迭代法求解，提高大规模网格计算性能
            self.U[:, k + 1], _ = gmres(A, F)

        # 计算误差
        x, t = np.meshgrid(self.x, self.t, indexing='ij')
        exact_solution = (x**2)*np.exp(t)
        self.MSE = np.mean((self.U - exact_solution) ** 2)

    def plot_results(self):
        # 生成数值解、精确解和误差的图像
        x, t = np.meshgrid(self.x, self.t, indexing='ij')
        exact_solution = (x**2)*np.exp(t)
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
        return 0  # u_x(0,t)=0

    def g2(t):
        return 2 * np.exp(t)  # u_x(1,t)=2e^t

    def u0(x):
        return x ** 2


    def f(x, t):
        return np.exp(t) * (4 * x**2 - 2)  # 根据方程推导调整

    def b(x, t):
        return 2 * x  # b(x,t)=2x

    def c(x, t):
        return -1  # c(x,t)=-1

    n, m = 50, 200
    solver = PDESolver(n, m, 0, 1, 0, 1, g1, g2, u0, b, c, f)
    solver.solve()
    solver.plot_results()
    print(f"均方误差 (MSE): {solver.MSE:.6e}")

