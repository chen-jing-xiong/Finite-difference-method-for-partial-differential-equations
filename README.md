# Finite-difference-method-of-partial-differential-equation
# 数值偏微分方程求解代码库

## 简介
本代码库主要包含了一系列用于数值求解偏微分方程（PDEs）和常微分方程（ODEs）的Python代码。通过有限差分方法（FDM），我们实现了多种类型的方程求解，包括一维和二维的热传导方程、抛物型方程、泊松方程以及一般的常微分方程，同时考虑了不同类型的边界条件，如狄利克雷（Dirichlet）和诺伊曼（Neumann）边界条件。

## 代码功能概述

### 1. 常微分方程求解（ODEs）
- **General_ODE_FDM.py**：
    - 实现了一个通用的常微分方程求解器类。
    - 支持自定义的边界条件和系数函数，包括 `b_x(x)` 和 `c_x(x)`。
    - 构建系数矩阵 `A` 和右端向量 `F`，并可通过线性方程组求解得到数值解。

### 2. 一维偏微分方程求解（1D PDEs）
- **1D_Heat_Dirichlet_FDM.py**：
    - 用于求解一维热传导方程，考虑狄利克雷边界条件。
    - 采用隐式差分格式，构建系数矩阵 `A` 并求解线性方程组。
- **1D_Heat_Neumann_FDM.py**：
    - 求解一维热传导方程，考虑诺伊曼边界条件。
    - 同样采用隐式差分格式，处理边界条件时对系数矩阵进行修正。
- **1D_general_Parabolic_Neumann_FDM.py**：
    - 求解一维一般抛物型方程，考虑诺伊曼边界条件。
    - 构建隐式差分格式的系数矩阵 `A`，并处理边界条件。

### 3. 二维偏微分方程求解（2D PDEs）
- **2D_general_Parabolic_Dirichlet_FDM.py**：
    - 求解二维一般抛物型方程，考虑狄利克雷边界条件。
    - 采用Crank–Nicolson格式，构建系数矩阵 `A` 并求解线性方程组。
    - 提供精确解计算和均方误差（MSE）计算功能，同时支持结果可视化。
- **2D_Passion_Dirichlet_FDM.py**：
    - 求解二维泊松方程，考虑狄利克雷边界条件。
    - 构建离散化后的系数矩阵 `A` 和右端向量 `F`，使用GMRES方法求解线性方程组。
- **2D_Passion_Neumann_FDM.py**：
    - 求解二维泊松方程，考虑诺伊曼边界条件。
    - 构建系数矩阵 `A` 和右端向量 `F`，处理边界条件时添加额外项。
- **2D_general_elliptic_Neumann_FDM.py**：
    - 求解二维一般椭圆型方程，考虑诺伊曼边界条件。
    - 构建系数矩阵 `A`，处理边界条件时对系数进行调整。

## 代码特点
- **模块化设计**：每个求解器类都具有清晰的结构，便于理解和扩展。
- **支持多种边界条件**：涵盖了狄利克雷和诺伊曼边界条件，可满足不同问题的需求。
- **可视化功能**：部分代码提供了结果可视化功能，方便直观地观察数值解和误差分布。

## 使用说明
每个求解器类的初始化函数都接受一系列参数，包括网格点数、边界条件、初始条件和系数函数等。通过调用相应的求解方法（如 `solve()`），可以得到数值解，并可进一步计算误差或进行可视化。

## 注意事项
本代码库中的代码均经过测试，确保其正确性。如果需要使用特定的方程或边界条件，只需根据需求修改相应的系数函数和边界条件函数即可。
# Numerical PDE Solver Code Repository

## Introduction
This code repository contains a series of Python codes for numerically solving partial differential equations (PDEs) and ordinary differential equations (ODEs). Using the finite difference method (FDM), we have implemented solvers for various types of equations, including one-dimensional and two-dimensional heat conduction equations, parabolic equations, Poisson equations, and general ordinary differential equations. Different types of boundary conditions, such as Dirichlet and Neumann boundary conditions, are also considered.

## Overview of Code Functions

### 1. Ordinary Differential Equation Solver (ODEs)
- **General_ODE_FDM.py**:
    - Implements a general ODE solver class.
    - Supports custom boundary conditions and coefficient functions, including `b_x(x)` and `c_x(x)`.
    - Constructs the coefficient matrix `A` and the right-hand side vector `F`, and obtains the numerical solution by solving a linear system of equations.

### 2. One-Dimensional Partial Differential Equation Solver (1D PDEs)
- **1D_Heat_Dirichlet_FDM.py**:
    - Solves the one-dimensional heat conduction equation with Dirichlet boundary conditions.
    - Uses an implicit difference scheme to construct the coefficient matrix `A` and solve the linear system of equations.
- **1D_Heat_Neumann_FDM.py**:
    - Solves the one-dimensional heat conduction equation with Neumann boundary conditions.
    - Also uses an implicit difference scheme and modifies the coefficient matrix when handling boundary conditions.
- **1D_general_Parabolic_Neumann_FDM.py**:
    - Solves the one-dimensional general parabolic equation with Neumann boundary conditions.
    - Constructs the coefficient matrix `A` of the implicit difference scheme and handles boundary conditions.

### 3. Two-Dimensional Partial Differential Equation Solver (2D PDEs)
- **2D_general_Parabolic_Dirichlet_FDM.py**:
    - Solves the two-dimensional general parabolic equation with Dirichlet boundary conditions.
    - Uses the Crank–Nicolson scheme to construct the coefficient matrix `A` and solve the linear system of equations.
    - Provides functions for calculating the exact solution and the mean squared error (MSE), and supports result visualization.
- **2D_Passion_Dirichlet_FDM.py**:
    - Solves the two-dimensional Poisson equation with Dirichlet boundary conditions.
    - Constructs the discretized coefficient matrix `A` and the right-hand side vector `F`, and uses the GMRES method to solve the linear system of equations.
- **2D_Passion_Neumann_FDM.py**:
    - Solves the two-dimensional Poisson equation with Neumann boundary conditions.
    - Constructs the coefficient matrix `A` and the right-hand side vector `F`, and adds additional terms when handling boundary conditions.
- **2D_general_elliptic_Neumann_FDM.py**:
    - Solves the two-dimensional general elliptic equation with Neumann boundary conditions.
    - Constructs the coefficient matrix `A` and adjusts the coefficients when handling boundary conditions.

## Code Features
- **Modular Design**: Each solver class has a clear structure, making it easy to understand and extend.
- **Support for Multiple Boundary Conditions**: Covers both Dirichlet and Neumann boundary conditions, meeting the needs of different problems.
- **Visualization Function**: Some codes provide result visualization functions, facilitating the intuitive observation of numerical solutions and error distributions.

## Usage Instructions
The initialization functions of each solver class accept a series of parameters, including the number of grid points, boundary conditions, initial conditions, and coefficient functions. By calling the corresponding solution methods (e.g., `solve()`), the numerical solution can be obtained, and further error calculations or visualizations can be performed.

## Notes
All the codes in this repository have been tested to ensure their correctness. If you need to use specific equations or boundary conditions, you only need to modify the corresponding coefficient functions and boundary condition functions according to your needs.
