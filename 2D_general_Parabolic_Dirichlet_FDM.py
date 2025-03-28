"""

二维 抛物线方程求解

方程形式：
    u_{t} -u_{xx}-u_{yy}+b(x,y,t)*u_{x}+c(x,y,t)u_{y}+d(x,y,t)u(x,y,t)=f(x,y,t),  a < x < b,c<y<d ,t>0

边界条件：
    u(a,y,t)=g1(t),  u(c,y,t)=g2(t)
    u(x,c,t)=g3(t),  u(x,d,t)=g4(t)
    u(x,y,0)=u0(x)

具体问题：
    u_{t} -u_{xx}-u_{yy}+u_{x}+u_{y}+u(x,y,t)=e^(-3t)sin(x+y)
    0 < x < pi,0<y<pi ,t>0

边界条件：
    u(a,y,t)=0,u(c,y,t)=0
    u(x,c,t)=0,u(x,d,t)=0
    u(x,y,0)=sin(x)sin(y)


精确解：
    u(x, y) = e^(-3t)sin(x)sin(y)

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import gmres


class PDESolver:
    def __init__(self, n, m, l, x_left_boundary, x_right_boundary, y_left_boundary, y_right_boundary, t_left_boundary,
                 t_right_boundary,
                 g1, g2, g3, g4, u0, b, c, d, f):
        # 初始化空间和时间离散参数--------------------------------
        self.xn = n                                                        # x坐标离散点数
        self.xh = (x_right_boundary - x_left_boundary) / n                 # x坐标步长
        self.x_left = x_left_boundary                                      # x坐标左端点
        self.x_right = x_right_boundary                                    # x坐标右端点
        self.x = np.linspace(x_left_boundary, x_right_boundary, n + 1)     # x坐标点

        self.yn = m                                                        # y坐标离散点数
        self.yh = (y_right_boundary - y_left_boundary) / m                 # y坐标步长
        self.y_left = y_left_boundary                                      # y坐标左端点
        self.y_right = y_right_boundary                                    # y坐标右端点
        self.y = np.linspace(y_left_boundary, y_right_boundary, m + 1)     # y坐标点

        self.tn = l                                                        # 时间离散点数
        self.th = (t_right_boundary - t_left_boundary) / l                 # 时间步长
        self.t_left = t_left_boundary                                      # 时间左端点
        self.t_right = t_right_boundary                                    # 时间右端点
        self.t = np.linspace(t_left_boundary, t_right_boundary, l + 1)     # 时间点

        # 初始化边界条件函数---------------------------------------
        self.g1 = g1        # 左边界函数
        self.g2 = g2        # 右边界函数
        self.g3 = g3        # 下边界函数
        self.g4 = g4        # 上边界函数
        self.u0 = u0        # 初始条件函数
        self.b = b          # u_x(x,y,t)系数
        self.c = c          # u_y(x,y,t)系数
        self.d = d          # u(x,y,t)系数
        self.f = f          # 源项函数

        # 初始化解和误差------------------------------------------
        self.U = np.zeros((self.tn + 1, (self.xn - 1) * (self.yn - 1)))   # 数值解矩阵
        self.exact = self.exact_solution()                                # 精确解矩阵
        self.MSE = None                                                   # 误差

    def build_A(self, n):
        """
        创建Crank–Nicolson格式的系数矩阵
        :param n: 第n个时间点
        :return:
        """

        xh, xh2 = self.xh, self.xh ** 2
        yh, yh2 = self.yh, self.yh ** 2
        th = self.th
        N = (self.xn - 1) * (self.yn - 1)
        A = lil_matrix((N, N))

        for i in range(self.xn - 1):
            for j in range(self.yn - 1):
                k = i * (self.yn - 1) + j
                A[k, k] = 1 / th + 1 / xh2 + 1 / yh2 + self.d(self.x[i + 1], self.y[j + 1], self.t[n + 1]) / 2
                if i > 0:
                    A[k, k - (self.yn - 1)] = -1 / (2 * xh2) - self.b(self.x[i + 1], self.y[j + 1], self.t[n + 1]) / (
                            4 * xh)
                if i < self.xn - 2:
                    A[k, k + (self.yn - 1)] = -1 / (2 * xh2) + self.b(self.x[i + 1], self.y[j + 1], self.t[n + 1]) / (
                            4 * xh)
                if j > 0:
                    A[k, k - 1] = -1 / (2 * yh2) - self.c(self.x[i + 1], self.y[j + 1], self.t[n + 1]) / (4 * yh)
                if j < self.yn - 2:
                    A[k, k + 1] = -1 / (2 * yh2) + self.c(self.x[i + 1], self.y[j + 1], self.t[n + 1]) / (4 * yh)

        return A

    def exact_solution(self):
        exact = np.zeros((self.tn + 1, (self.xn - 1) * (self.yn - 1)))
        for n in range(self.tn + 1):
            for i in range(self.xn - 1):
                for j in range(self.yn - 1):
                    k = i * (self.yn - 1) + j
                    exact[n, k] = np.exp(-3 * self.t[n]) * np.sin(self.x[i + 1]) * np.sin(self.y[j + 1])
        return exact

    def solve(self):
        xh, xh2 = self.xh, self.xh ** 2
        yh, yh2 = self.yh, self.yh ** 2
        th = self.th

        # 设置初始条件
        for i in range(self.xn - 1):
            for j in range(self.yn - 1):
                k = i * (self.yn - 1) + j
                self.U[0, k] = self.u0(self.x[i + 1], self.y[j + 1], 0)

        F = np.zeros((self.xn - 1) * (self.yn - 1))
        for n in range(self.tn):
            for i in range(self.xn - 1):
                for j in range(self.yn - 1):
                    k = i * (self.yn - 1) + j
                    F[k] = ((self.f(self.x[i + 1], self.y[j + 1], self.t[n + 1]) + self.f(self.x[i + 1], self.y[j + 1],
                                                                                          self.t[n])) / 2
                            - (1 / xh2 + 1 / yh2 + self.d(self.x[i + 1], self.y[j + 1], self.t[n]) / 2) * self.U[
                                n, k]) + self.U[n, k] / th

                    # 修正邻居节点的贡献
                    if i > 0:  # 左邻居
                        F[k] += (1 / (2 * xh2) + self.b(self.x[i + 1], self.y[j + 1], self.t[n]) / (4 * xh)) * self.U[
                            n, k - (self.yn - 1)]
                    if i < self.xn - 2:  # 右邻居
                        F[k] += (1 / (2 * xh2) - self.b(self.x[i + 1], self.y[j + 1], self.t[n]) / (4 * xh)) * self.U[
                            n, k + (self.yn - 1)]
                    if j > 0:  # 下邻居
                        F[k] += (1 / (2 * yh2) + self.c(self.x[i + 1], self.y[j + 1], self.t[n]) / (4 * yh)) * self.U[
                            n, k - 1]
                    if j < self.yn - 2:  # 上邻居
                        F[k] += (1 / (2 * yh2) - self.c(self.x[i + 1], self.y[j + 1], self.t[n]) / (4 * yh)) * self.U[
                            n, k + 1]

                    # 处理边界条件
                    if i == 0: # 左边界
                        F[k] -= (-1 / (2 * xh2) - self.b(self.x[i + 1], self.y[j + 1], self.t[n + 1]) / (
                                4 * xh)) * self.g1(self.x[i], self.y[j + 1], self.t[n + 1])
                        F[k] += (1 / (2 * xh2) + self.b(self.x[i + 1], self.y[j + 1], self.t[n]) / (4 * xh)) * self.g1(
                            self.x[i], self.y[j + 1], self.t[n])
                    if i == self.xn - 2: # 右边界
                        F[k] -= (-1 / (2 * xh2) + self.b(self.x[i + 1], self.y[j + 1], self.t[n + 1]) / (
                                4 * xh)) * self.g2(self.x[i + 2], self.y[j + 1], self.t[n + 1])
                        F[k] += (1 / (2 * xh2) - self.b(self.x[i + 1], self.y[j + 1], self.t[n]) / (4 * xh)) * self.g2(
                            self.x[i + 2], self.y[j + 1], self.t[n])
                    if j == 0:   # 下边界
                        F[k] -= (-1 / (2 * yh2) - self.c(self.x[i + 1], self.y[j + 1], self.t[n + 1]) / (
                                4 * yh)) * self.g3(self.x[i + 1], self.y[j], self.t[n + 1])
                        F[k] += (1 / (2 * yh2) + self.c(self.x[i + 1], self.y[j + 1], self.t[n]) / (4 * yh)) * self.g3(
                            self.x[i + 1], self.y[j], self.t[n])
                    if j == self.yn - 2: # 上边界
                        F[k] -= (-1 / (2 * yh2) + self.c(self.x[i + 1], self.y[j + 1], self.t[n + 1]) / (
                                4 * yh)) * self.g4(self.x[i + 1], self.y[j + 2], self.t[n + 1])
                        F[k] += (1 / (2 * yh2) - self.c(self.x[i + 1], self.y[j + 1], self.t[n]) / (4 * yh)) * self.g4(
                            self.x[i + 1], self.y[j + 2], self.t[n])

            A = self.build_A(n) # 构建A矩阵
            self.U[n + 1, :], _ = gmres(A, F) # 求解线性方程组，得到U[n+1,:]

        self.MSE = np.mean((self.U - self.exact_solution()) ** 2) # 计算MSE

    def plot_results(self):
        x, y = np.meshgrid(self.x[1:-1], self.y[1:-1], indexing='ij')

        # 创建图形
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # 初始化三个子图的等高线图
        numerical_solution = self.U[0].reshape(self.xn - 1, self.yn - 1)
        exact_solution = np.exp(-3 * self.t[0]) * np.sin(x) * np.sin(y)
        error = np.abs(numerical_solution - exact_solution)

        cp1 = ax[0].contourf(x, y, numerical_solution, 20, cmap='viridis')
        cp2 = ax[1].contourf(x, y, exact_solution, 20, cmap='viridis')
        cp3 = ax[2].contourf(x, y, error, 20, cmap='viridis')

        # 添加颜色条
        fig.colorbar(cp1, ax=ax[0])
        fig.colorbar(cp2, ax=ax[1])
        # fig.colorbar(cp3, ax=ax[2])

        # 设置标题
        ax[0].set_title('Numerical Solution')
        ax[1].set_title('Exact Solution')
        ax[2].set_title('Error')

        # 添加时间文本
        time_text = fig.suptitle(f'Time: {self.t[0]:.3f}')

        plt.tight_layout()

        # 更新函数
        def update(frame):
            # 更新数值解
            numerical_solution = self.U[frame].reshape(self.xn - 1, self.yn - 1)
            exact_solution = np.exp(-3 * self.t[frame]) * np.sin(x) * np.sin(y)
            error = np.abs(numerical_solution - exact_solution)

            # 清除旧的等高线图
            for coll in ax[0].collections:
                coll.remove()
            for coll in ax[1].collections:
                coll.remove()
            for coll in ax[2].collections:
                coll.remove()

            # 绘制新的等高线图
            ax[0].contourf(x, y, numerical_solution, 20, cmap='viridis')
            ax[1].contourf(x, y, exact_solution, 20, cmap='viridis')
            ax[2].contourf(x, y, error, 20, cmap='viridis')

            # 更新时间标题
            time_text.set_text(f'Time: {self.t[frame]:.3f}')

            return ax[0], ax[1], ax[2], time_text

        # 创建动画
        anim = FuncAnimation(fig, update, frames=range(len(self.t)),
                             interval=100, blit=False)

        plt.show()

        return anim


if __name__ == "__main__":
    # 定义边界条件函数
    def g1(x, y, t): return 0
    def g2(x, y, t): return 0
    def g3(x, y, t): return 0
    def g4(x, y, t): return 0
    def b(x, y, t): return 1
    def d(x, y, t): return 1
    def c(x, y, t): return 1
    def u0(x, y, t): return np.sin(x) * np.sin(y)
    def f(x, y, t): return np.exp(-3 * t) * np.sin(x + y)


    n, m, l = 20, 20, 100
    solver = PDESolver(n, m, l, 0, np.pi, 0, np.pi, 0, 1, g1, g2, g3, g4, u0, b, c, d, f)
    solver.solve()
    solver.plot_results()
    print(f"均方误差 (MSE): {solver.MSE:.6e}")
