"""
泊松方程
u_xx+u_yy=f(x,y)   a<x<b,c<y<d
$$
\frac{\partial u\left( x,y \right)}{\partial n}=g(x,y),on\partial \varOmega
$$

u_xx+u_yy=-2*pi^2 * sin(pi * x) * cos(pi * y)  0<x<1,0<y<1

边界条件
u_x(x,0)=0,u_x(x,1)=0
u_y(0,y)=pi*cos(pi*y),u_y(1,y)=pi * cos(pi * y)

精确解 u(x,y)=np.sin(np.pi*X) * np.cos(np.pi * Y)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import gmres


class PDESolver:
    def __init__(self, n, m, x_left_boundary, x_right_boundary, y_left_boundary, y_right_boundary, u1, u2, u3, u4, f):
        self.xn = n
        self.xh = (x_right_boundary - x_left_boundary) / n
        self.x_left = x_left_boundary
        self.x_right = x_right_boundary
        self.x = np.linspace(x_left_boundary, x_right_boundary, n + 1)

        self.yn = m
        self.yh = (y_right_boundary - y_left_boundary) / m
        self.y_left = y_left_boundary
        self.y_right = y_right_boundary
        self.y = np.linspace(y_left_boundary, y_right_boundary, m + 1)

        self.u1 = u1  # 左边界
        self.u2 = u2  # 右边界
        self.u3 = u3  # 下边界
        self.u4 = u4  # 上边界
        self.f = f

        self.A = self.build_A()
        self.F = self.build_F()
        self.U = None

    def build_A(self):
        N = (self.xn + 1) * (self.yn + 1)
        A = lil_matrix((N, N))
        hx, hy = self.xh, self.yh
        hx2, hy2 = hx ** 2, hy ** 2

        for i in range(self.xn + 1):
            for j in range(self.yn + 1):
                k = i * (self.yn + 1) + j
                A[k,k]= -2 / hx2 - 2 / hy2

                if i == 0:  # 左边界Neumann
                    A[k, k + (self.yn + 1)] = 2 / hx2  # 右邻点
                elif i == self.xn:  # 右边界Neumann
                    A[k, k - (self.yn + 1)] = 2 / hx2  # 左邻点
                else:  # 内部点
                    A[k, k + (self.yn + 1)] = 1 / hx2
                    A[k, k - (self.yn + 1)] = 1 / hx2

                if j == 0:  # 下边界Neumann
                    A[k, k + 1] = 2 / hy2  # 上邻点

                elif j == self.yn:  # 上边界Neumann
                    A[k, k - 1] = 2 / hy2  # 下邻点

                else:  # 内部点
                    A[k, k + 1] = 1 / hy2
                    A[k, k - 1] = 1 / hy2


        return A.tocsr()

    def build_F(self):
        F = np.zeros((self.xn + 1) * (self.yn + 1))
        hx, hy = self.xh, self.yh

        for i in range(self.xn + 1):
            for j in range(self.yn + 1):
                k = i * (self.yn + 1) + j
                F[k] = self.f(self.x[i], self.y[j])

                # 添加Neumann边界项
                if i == 0:  # 左边界
                    F[k] += 2 * self.u1(self.x[i], self.y[j]) / hx
                elif i == self.xn:  # 右边界
                    F[k] -= 2 * self.u2(self.x[i], self.y[j]) / hx
                if j == 0:  # 下边界
                    F[k] += 2 * self.u3(self.x[i], self.y[j]) / hy
                elif j == self.yn:  # 上边界
                    F[k] -= 2 * self.u4(self.x[i], self.y[j]) / hy

        return F

    def solve(self):
        self.U, _ = gmres(self.A, self.F, atol=1e-12)

    def plot_results(self):
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        # 重塑解向量为二维数组
        U_2d = self.U.reshape((self.xn+1, self.yn+1))
        # 计算精确解（示例用指数正弦函数，需根据实际问题修改）
        exact_solution = np.sin(np.pi*X) * np.cos(np.pi * Y)
        # 计算误差
        error = np.abs(U_2d - exact_solution)

        fig = plt.figure(figsize=(18, 5))

        # 绘制数值解
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(X, Y, U_2d, cmap='viridis')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Numerical Solution')

        # 绘制精确解
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(X, Y, exact_solution, cmap='viridis')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Exact Solution')

        # 绘制误差图
        ax3 = fig.add_subplot(133)
        im = ax3.imshow(error, cmap='viridis', origin='lower',
                        extent=[self.x[1], self.x[-2], self.y[1], self.y[-2]])
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('Error')
        fig.colorbar(im, ax=ax3)  # 添加颜色条

        plt.show()


# 正确边界条件定义（注意符号调
if __name__ == "__main__":
    def u1(x, y):
        return np.pi * np.cos(np.pi * y)


    def u2(x, y):
        return -np.pi * np.cos(np.pi * y)

    def u3(x, y):
        return 0.0

    def u4(x, y):
        return 0.0

    def f(x, y):
        return -2 * (np.pi ** 2) * np.sin(np.pi * x) * np.cos(np.pi * y)


    n, m = 30, 30


    solver = PDESolver(n, m, 0, 1, 0, 1, u1, u2, u3, u4, f)
    solver.solve()
    solver.plot_results()

    x, y = np.meshgrid(solver.x, solver.y, indexing='ij')
    exact = np.sin(np.pi*x) * np.cos(np.pi * y)
    U_2d = solver.U.reshape((solver.xn + 1, solver.yn + 1))
    error = np.mean((U_2d - exact)**2)
    print(f"相对误差: ",error)
    # np.set_printoptions(linewidth=10000,)
    # dense_A = solver.A.toarray()
    # print("转换为密集矩阵后的 A:")
    # print(dense_A)

