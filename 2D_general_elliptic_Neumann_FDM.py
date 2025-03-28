"""
泊松方程求解

方程形式：
    u_{xx} + u_{yy} + (x+y)u_x + (x-y)u_y + (1+x^2+y^2)u = f(x,y),  0 < x < 1, 0 < y < 1

边界条件：
    u_x(x, 0) = x,        u_x(x, 1) = -x
    u_y(0, y) = y,        u_y(1, y) = -y

精确解：
    u(x, y) = x^2 + y^2
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import gmres


class PDESolver:
    def __init__(self, n, m, x_left_boundary, x_right_boundary, y_left_boundary, y_right_boundary, u1, u2, u3, u4, b,c,f):
        self.xn = n                                                     # 划分网格数
        self.xh = (x_right_boundary - x_left_boundary) / n              # 网格单元长度
        self.x_left = x_left_boundary                                   # 左边界
        self.x_right = x_right_boundary                                 # 右边界
        self.x = np.linspace(x_left_boundary, x_right_boundary, n + 1)  # x坐标划分点向量

        self.yn = m
        self.yh = (y_right_boundary - y_left_boundary) / m
        self.y_left = y_left_boundary
        self.y_right = y_right_boundary
        self.y = np.linspace(y_left_boundary, y_right_boundary, m + 1)

        self.u1 = u1  # 左边界导数函数
        self.u2 = u2  # 右边界导数函数
        self.u3 = u3  # 下边界导数函数
        self.u4 = u4  # 上边界导数函数
        self.f = f    #右端函数
        self.b=b      #一阶偏导u_x,u_y系数
        self.c=c      #u(x,y)系数

        self.A = self.build_A() # 构建系数A矩阵
        self.F = self.build_F() # 构建右端向量F
        self.U = None           # 数值解
        self.MSE=None           #均方误差

    def build_A(self):
        N = (self.xn + 1) * (self.yn + 1)
        A = lil_matrix((N, N))
        hx, hy = self.xh, self.yh
        hx2, hy2 = hx ** 2, hy ** 2

        for i in range(self.xn + 1):
            for j in range(self.yn + 1):
                k = i * (self.yn + 1) + j
                A[k,k]= -2 / hx2 - 2 / hy2+self.c(self.x[i],self.y[j])

                if i == 0:  # 左边界Neumann
                    A[k, k + (self.yn + 1)] = 2 / hx2  # 右邻点
                elif i == self.xn:  # 右边界Neumann
                    A[k, k - (self.yn + 1)] = 2 / hx2  # 左邻点
                else:  # 内部点
                    A[k, k + (self.yn + 1)] = 1 / hx2+self.b(self.x[i],self.y[j])/(2*hx)
                    A[k, k - (self.yn + 1)] = 1 / hx2-self.b(self.x[i],self.y[j])/(2*hx)

                if j == 0:  # 下边界Neumann
                    A[k, k + 1] = 2 / hy2  # 上邻点

                elif j == self.yn:  # 上边界Neumann
                    A[k, k - 1] = 2 / hy2  # 下邻点

                else:  # 内部点
                    A[k, k + 1] = 1 / hy2+self.b(self.x[i],self.y[j])/(2*hy)
                    A[k, k - 1] = 1 / hy2-self.b(self.x[i],self.y[j])/(2*hy)


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
                    F[k] += 2 * self.u1(self.x[i], self.y[j]) / hx-self.b(self.x[i],self.y[j])* self.u1(self.x[i], self.y[j])
                elif i == self.xn:  # 右边界
                    F[k] -= 2 * self.u2(self.x[i], self.y[j]) / hx+self.b(self.x[i],self.y[j])* self.u2(self.x[i], self.y[j])
                if j == 0:  # 下边界
                    F[k] += 2 * self.u3(self.x[i], self.y[j]) / hy-self.b(self.x[i],self.y[j])* self.u3(self.x[i], self.y[j])
                elif j == self.yn:  # 上边界
                    F[k] -= 2 * self.u4(self.x[i], self.y[j]) / hy+self.b(self.x[i],self.y[j])* self.u4(self.x[i], self.y[j])

        return F

    def solve(self):
        self.U, _ = gmres(self.A, self.F, atol=1e-8)
        x, y = np.meshgrid(self.x, self.y, indexing='ij')
        U_2d = self.U.reshape((self.xn + 1, self.yn + 1))
        exact_solution = np.sin(np.pi*x) + np.sin(np.pi * y)
        self.MSE = np.mean((exact_solution - U_2d) ** 2)

    def plot_results(self):
        x, y = np.meshgrid(self.x, self.y, indexing='ij')
        U_2d = self.U.reshape((self.xn+1, self.yn+1))
        # 精确解
        exact_solution = np.sin(np.pi*x) + np.sin(np.pi * y)
        # 计算误差
        error = U_2d - exact_solution


        fig = plt.figure(figsize=(18, 5))

        # 绘制数值解
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(x, y, U_2d, cmap='viridis')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Numerical Solution')

        # 绘制精确解
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(x, y, exact_solution, cmap='viridis')
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



if __name__ == "__main__":
    def u1(x, y):
        return np.pi


    def u2(x, y):
        return -np.pi


    def u3(x, y):
        return np.pi


    def u4(x, y):
        return -np.pi


    def f(x, y):
        term = (-np.pi ** 2 + 2) * (np.sin(np.pi * x) + np.sin(np.pi * y))
        term += np.pi * (np.cos(np.pi * x) + np.cos(np.pi * y))
        return term


    def b(x, y):
        return 1.0


    def c(x, y):
        return 2.0


    n, m = 30, 30  # 网格划分

    solver = PDESolver(n, m, 0, 1, 0, 1, u1, u2, u3, u4, b, c, f)
    solver.solve()
    solver.plot_results()
    print(f"均方误差: ", solver.MSE)

