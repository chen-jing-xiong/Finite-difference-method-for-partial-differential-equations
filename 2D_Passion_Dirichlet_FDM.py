"""
泊松方程
u_xx+u_yy=f(x,y)   a<x<b,c<y<d
u(x,y)=g(x,y)  on \partial \varOmega


u_xx+u_yy=(1-pi^2) * e^(x) * sin(pi * y)  0<x<2,0<y<1

边界条件
u(x,0)=0,u(x,1)=0
u(0,y)=sin(pi*y),u(2,y)=e^(2) * sin(pi * y)

精确解 u(x,y)=e^(x) * sin(pi * y)
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import gmres

class PDESolver:
    def __init__(self, n, m, x_left_boundary, x_right_boundary, y_left_boundary, y_right_boundary, u1, u2, u3, u4, f):
        """
        二维泊松方程
        u_xx + u_yy = f(x,y)  在矩形区域 a < x < b, c < y < d 内
        边界条件：
        u(a,y) = u1(y)
        u(b,y) = u2(y)
        u(x,c) = u3(x)
        u(x,d) = u4(x)

        参数:
        :param n: x方向网格划分数（内部节点数）
        :param m: y方向网格划分数（内部节点数）
        :param x_left_boundary: x方向左边界
        :param x_right_boundary: x方向右边界
        :param y_left_boundary: y方向下边界
        :param y_right_boundary: y方向上边界
        :param u1: 左边界条件函数
        :param u2: 右边界条件函数
        :param u3: 下边界条件函数
        :param u4: 上边界条件函数
        :param f: 方程右端项函数
        """
        # x方向参数初始化
        self.xn = n  # x方向网格划分数
        self.xh = (x_right_boundary - x_left_boundary) / n  # x方向步长
        self.x_left = x_left_boundary
        self.x_right = x_right_boundary
        self.x = self.generate_grid(self.x_left, self.xh, self.xn)  # 生成x方向网格点

        # y方向参数初始化
        self.yn = m  # y方向网格划分数
        self.yh = (y_right_boundary - y_left_boundary) / m  # y方向步长
        self.y_left = y_left_boundary
        self.y_right = y_right_boundary
        self.y = self.generate_grid(self.y_left, self.yh, self.yn)  # 生成y方向网格点

        # 边界条件函数
        self.u1 = u1  # 左边界
        self.u2 = u2  # 右边界
        self.u3 = u3  # 下边界
        self.u4 = u4  # 上边界

        self.f = f  # 右端项函数
        self.A = self.build_A()  # 系数矩阵
        self.F = self.build_F()  # 右端向量
        self.U = None  # 数值解存储

    def generate_grid(self, a, h, n):
        """
        生成网格点数组
        参数:
        :param a: 起始点
        :param h: 步长
        :param n: 划分数（内部节点数）
        返回:
        :return: 包含网格点的数组（n+1个点）
        """
        return np.linspace(a, a + n * h, n + 1)

    def build_A(self):
        """
        构建离散化后的系数矩阵A
        使用五点差分格式，处理稀疏矩阵
        返回:
        :return: CSR格式的稀疏矩阵
        """
        N = (self.xn - 1) * (self.yn - 1)  # 内部节点总数
        A = lil_matrix((N, N))  # 初始化稀疏矩阵
        hx2 = self.xh ** 2  # x方向步长平方
        hy2 = self.yh ** 2  # y方向步长平方

        for i in range(self.xn - 1):  # 遍历x方向网格
            for j in range(self.yn - 1):  # 遍历y方向网格
                k = i * (self.yn - 1) + j  # 当前节点的全局索引
                # 中心节点系数（五点差分核心项）
                A[k, k] = -2 * (1 / hx2 + 1 / hy2)

                # 处理x方向邻接节点
                if i > 0:  # 存在左边节点
                    A[k, k - (self.yn - 1)] = 1 / hx2  # 左节点系数
                if i < self.xn - 2:  # 存在右边节点
                    A[k, k + (self.yn - 1)] = 1 / hx2  # 右节点系数

                # 处理y方向邻接节点
                if j > 0:  # 存在下边节点
                    A[k, k - 1] = 1 / hy2  # 下节点系数
                if j < self.yn - 2:  # 存在上边节点
                    A[k, k + 1] = 1 / hy2  # 上节点系数

        return A.tocsr()  # 转换为压缩行格式，便于求解

    def build_F(self):
        """
        构建线性方程组的右端向量F
        包含内部节点的源项和边界条件的影响
        返回:
        :return: 右端向量F
        """
        F = np.zeros((self.xn - 1) * (self.yn - 1))  # 初始化右端向量
        for i in range(self.xn - 1):
            for j in range(self.yn - 1):
                k = i * (self.yn - 1) + j
                # 计算内部节点的源项
                F[k] = self.f(self.x[i + 1], self.y[j + 1])

                # 处理左边界条件
                if i == 0:
                    F[k] -= self.u1(self.x[i], self.y[j + 1]) / self.xh ** 2
                # 处理右边界条件
                if i == self.xn - 2:
                    F[k] -= self.u2(self.x[i + 2], self.y[j + 1]) / self.xh ** 2
                # 处理下边界条件
                if j == 0:
                    F[k] -= self.u3(self.x[i + 1], self.y[j]) / self.yh ** 2
                # 处理上边界条件
                if j == self.yn - 2:
                    F[k] -= self.u4(self.x[i + 1], self.y[j + 2]) / self.yh ** 2

        return F

    def solve(self):
        """
        使用GMRES方法求解线性方程组
        结果存储在self.U中
        """
        self.U, _ = gmres(self.A, self.F, atol=1e-8, maxiter=1000)

    def plot_results(self):
        """
        绘制数值解、精确解和误差的三维图形
        """
        # 生成网格坐标矩阵
        X, Y = np.meshgrid(self.x[1:-1], self.y[1:-1], indexing='ij')
        # 重塑解向量为二维数组
        U_2d = self.U.reshape((self.xn - 1, self.yn - 1))
        # 计算精确解（示例用指数正弦函数，需根据实际问题修改）
        exact_solution = np.exp(X) * np.sin(np.pi * Y)
        # 计算误差
        error = np.abs(U_2d - exact_solution)

        # 创建图形窗口
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



if __name__ == "__main__":

    n = 100
    m = 100
    x_left_boundary = 0
    x_right_boundary = 2
    y_left_boundary = 0
    y_right_boundary = 1

    def u1(x, y):
        return  np.sin(np.pi*y)

    def u2(x, y):
        return np.exp(2)*np.sin(np.pi*y)

    def u3(x, y):
        return 0

    def u4(x, y):
        return 0

    def f(x, y):
        return (1-np.pi ** 2) * np.exp(x) * np.sin(np.pi * y)

    solver = PDESolver(n, m, x_left_boundary, x_right_boundary, y_left_boundary, y_right_boundary, u1, u2, u3, u4, f)
    solver.solve()
    solver.plot_results()
    X, Y = np.meshgrid(solver.x[1:-1], solver.y[1:-1], indexing='ij')
    exact_solution = np.exp(X) * np.sin(np.pi * Y)
    U_2d = solver.U.reshape((solver.xn - 1, solver.yn - 1))
    error = np.mean((U_2d - exact_solution)**2)
    print("误差为：", error)
    # np.set_printoptions(linewidth=10000)
    # dense_A = solver.A.toarray()
    # print("转换为密集矩阵后的 A:")
    # print(dense_A)
