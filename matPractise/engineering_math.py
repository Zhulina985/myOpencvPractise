import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols

def task1():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, exp, sin, pi

    # ===================== 1. 问题定义与参数设置 =====================
    # 一阶线性双曲型PDE（传输方程）：u_t + c*u_x = 0
    # 描述波以恒定速度c沿x轴正方向传播，无衰减
    # 空间域：x ∈ [0, L]
    # 时间域：t ∈ [0, T_max]

    # 物理参数
    c = 1.0  # 波速（m/s），决定传播速度
    L = 10.0  # 空间域长度（m）
    T_max = 5.0  # 最大模拟时间（s）

    # 数值计算参数
    Nx = 200  # 空间离散点数
    Nt = 50  # 时间离散点数（用于最终时刻计算）

    # 生成空间网格和时间点
    x = np.linspace(0, L, Nx)
    t_final = T_max  # 最终时刻

    # ===================== 2. 初始条件与边界条件 =====================
    # --- 初始条件 (ICs)：t=0时的波形 ---
    # 采用高斯脉冲初始条件：u(x, 0) = exp(-(x-3)^2)
    u0 = np.exp(-(x - 3) ** 2)
    # 也可选择正弦波初始条件：u0 = np.sin(pi*(x-2)/4)（局部正弦脉冲）

    # --- 边界条件 (BCs)：周期性边界（可选，因传输方程无边界反射） ---
    # BC: u(0,t) = u(L,t)

    # ===================== 3. 分离变量法/行波解求解 =====================
    # 传输方程的解析解为行波解：u(x,t) = u0(x - c*t)（右行波）
    # 分离变量假设：u(x,t) = X(x)T(t)，代入方程得：X'/X = -T'/(cT) = -k（常数）
    # 解得：X(x) = A*exp(-k*x), T(t) = B*exp(-c*k*t) → u(x,t) = C*exp(-k(x - c*t))，与行波解一致

    # 计算初始时刻和最终时刻的解
    u_initial = u0.copy()
    u_final = np.exp(-(x - c * t_final - 3) ** 2)  # 最终时刻行波解

    # ===================== 4. 输出解析解表达式 =====================
    print("=" * 80)
    print("一阶传输方程解析解（行波解）：")
    x_sym, t_sym = symbols('x t')
    # 根据初始条件构造符号表达式
    if np.all(u0 == np.exp(-(x - 3) ** 2)):
        expr = f"u(x,t) = exp(-(x - {c}*t - 3)^2)"
    else:
        expr = f"u(x,t) = sin(pi*(x - {c}*t - 2)/4)"
    print(expr)
    print(f"物理意义：波形以速度{c}m/s沿x轴正方向传播，形状保持不变")
    print("=" * 80)

    # ===================== 5. 结果可视化 =====================
    plt.figure(figsize=(10, 6))

    # 绘制初始时刻和最终时刻的波形
    plt.plot(x, u_initial, label=f'Initial Time (t=0)', color='blue', linewidth=2)
    plt.plot(x, u_final, label=f'Final Time (t={t_final}s)', color='red', linewidth=2, linestyle='--')

    # 标注波传播方向
    plt.arrow(4, 0.8, c * t_final, 0, head_width=0.05, head_length=0.2, fc='green', ec='green',
              label='Wave Propagation Direction')

    # 图表设置
    plt.xlabel('Position x (m)', fontsize=12)
    plt.ylabel('Amplitude u(x,t)', fontsize=12)
    plt.title('1st-order Hyperbolic PDE (Transport Equation) Solution', fontsize=14)
    plt.xlim(0, L)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

    # ===================== 6. 结果验证 =====================
    # 验证波形传播距离：初始中心在x=3，最终中心应在x=3+c*t_final
    initial_center = x[np.argmax(u_initial)]
    final_center = x[np.argmax(u_final)]
    print(f"\n初始波形中心位置：x={initial_center:.2f}m")
    print(f"最终波形中心位置：x={final_center:.2f}m")
    print(f"理论传播距离：{c * t_final:.2f}m，实际传播距离：{final_center - initial_center:.2f}m（一致）")
    print(f"波形最大值验证：初始={np.max(u_initial):.4f}，最终={np.max(u_final):.4f}（无衰减）")

def task2_1d():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, sin, exp, pi

    # ===================== 1. 问题定义与参数设置 =====================
    # 1D热方程：u_t = k * u_xx （描述细杆的热传导过程）
    # 空间域：x ∈ [0, L]，时间域：t ∈ [0, T_max]

    # 物理参数
    k = 0.01  # 热扩散系数（m²/s），决定热量扩散速度
    L = 1.0  # 细杆长度（m）
    T_max = 1.0  # 模拟总时间（s）

    # 数值计算参数
    Nx = 100  # 空间离散点数（越多越精细）
    N_terms = 5  # 傅里叶级数项数（用于表达式显示）

    # 空间网格生成
    x = np.linspace(0, L, Nx)  # 从0到L均匀分布的空间点

    # ===================== 2. 初始条件(ICs)与边界条件(BCs) =====================
    # --- 初始条件 (ICs)：t=0时的温度分布 ---
    # 设初始温度为抛物线分布：u(x, 0) = x*(1-x)，中心温度最高
    u0 = x * (1 - x)

    # --- 边界条件 (BCs)：齐次Dirichlet边界（两端固定为0℃） ---
    # BC1: u(0, t) = 0 （x=0处温度始终为0）
    # BC2: u(L, t) = 0 （x=L处温度始终为0）

    # ===================== 3. 分离变量法求解 =====================
    # 分离变量假设：u(x,t) = X(x)*T(t)，代入热方程得：
    # X'' + λX = 0 → X_n(x) = sin(nπx/L)（满足BCs）
    # T' + kλT = 0 → T_n(t) = exp(-k(nπ/L)² t)
    # 通解：u(x,t) = Σ [Cn * sin(nπx/L) * exp(-k(nπ/L)² t)]
    # 其中Cn为傅里叶系数：Cn = (2/L)∫₀ᴸ u0(x)sin(nπx/L)dx

    # 存储初始时刻和最终时刻的解
    u_initial = u0.copy()
    u_final = np.zeros_like(x)

    # 存储级数项表达式和系数
    expression_terms = []
    Cn_list = []

    # 符号变量（用于表达式生成）
    x_sym, t_sym = symbols('x t')

    # 计算每一项级数并叠加
    for n in range(1, N_terms + 1):
        # 步骤1：计算傅里叶系数Cn（由初始条件确定）
        integrand = u0 * np.sin(n * np.pi * x / L)  # 被积函数
        Cn = (2 / L) * np.trapz(integrand, x)  # 梯形法数值积分
        Cn_list.append(round(Cn, 6))  # 保留6位小数

        # 步骤2：计算最终时刻的级数项
        lambda_n = (n * np.pi / L) ** 2  # 特征值λ_n
        T_n = np.exp(-k * lambda_n * T_max)  # 时间项在t=T_max时的值
        X_n = np.sin(n * np.pi * x / L)  # 空间项
        u_final += Cn * X_n * T_n  # 叠加到最终解

        # 步骤3：构造符号表达式（用于输出）
        term_expr = f"{Cn_list[-1]}*sin({n}*pi*x/{L})*exp(-{k}*({n}*pi/{L})²*t)"
        expression_terms.append(term_expr)

    # ===================== 4. 输出函数表达式 =====================
    print("=" * 80)
    print(f"1D热方程解的近似表达式（前{N_terms}项）：")
    print("u(x,t) = " + " + ".join(expression_terms))

    # ===================== 5. 结果可视化 =====================
    plt.figure(figsize=(10, 6))

    # 绘制初始时刻和最终时刻的温度分布
    plt.plot(x, u_initial, label=f'start (t=0)', color='blue', linewidth=2)
    plt.plot(x, u_final, label=f'final (t={T_max}s)', color='red', linewidth=2, linestyle='--')

    # 标注边界条件（两端温度为0）
    plt.scatter([0, L], [0, 0], color='black', s=50, label='边界条件 u(0,t)=u(L,t)=0')

    # 图表美化
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('u(x,t)', fontsize=12)
    plt.title('1D', fontsize=14)
    plt.xlim(0, L)
    plt.ylim(0, np.max(u_initial) * 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

    # ===================== 6. 结果验证 =====================
    print(f"\n初始时刻温度最大值：{np.max(u_initial):.4f}")
    print(f"最终时刻温度最大值：{np.max(u_final):.4f}")
    print(f"边界条件验证：u(0, T_max) = {u_final[0]:.6f}, u(L, T_max) = {u_final[-1]:.6f}（理论应为0）")
    print(f"温度衰减比例：{np.max(u_final) / np.max(u_initial):.4f}（体现热扩散的衰减特性）")

def task2_2d():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, sin, exp, pi
    from mpl_toolkits.mplot3d import Axes3D

    # ===================== 2D热方程求解 =====================
    # 参数设置
    k = 0.01  # 热扩散系数
    a = 1.0  # x方向长度
    b = 1.0  # y方向长度
    T_max = 1.0  # 最大时间
    Nx = 50  # x方向离散点数
    Ny = 50  # y方向离散点数
    N_terms = 5  # 级数项数

    # 生成网格
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    X, Y = np.meshgrid(x, y)

    # 初始条件
    u0 = np.sin(np.pi * X / a) * np.sin(np.pi * Y / b)

    # 最终时刻解
    u_final = np.zeros_like(X)
    expression_terms = []

    # 符号变量
    x_sym, y_sym, t_sym = symbols('x y t')

    # 分离变量法求解
    for m in range(1, N_terms + 1):
        for n in range(1, N_terms + 1):
            # 特征值
            lambda_mn = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2

            # 初始条件系数（此处初始条件为sin(πx/a)sin(πy/b)，仅m=1,n=1时有值）
            if m == 1 and n == 1:
                Cmn = 1.0
            else:
                Cmn = 0.0

            # 最终时刻解
            term = Cmn * np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b) * np.exp(-k * lambda_mn * T_max)
            u_final += term

            # 构造表达式
            if Cmn != 0:
                expr = f"{Cmn}*sin({m}*pi*x/{a})*sin({n}*pi*y/{b})*exp(-{k}*({m}²π²/{a}²+{n}²π²/{b}²)*t)"
                expression_terms.append(expr)

    # ===================== 输出表达式 =====================
    print("=" * 80)
    print("2D热方程解的表达式：")
    print("u(x,y,t) = " + " + ".join(expression_terms))
    print("=" * 80)

    # ===================== 可视化结果 =====================
    fig = plt.figure(figsize=(12, 5))

    # 初始时刻
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u0, cmap='viridis')
    ax1.set_title(f' (t=0)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')

    # 最终时刻
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_final, cmap='viridis')
    ax2.set_title(f' (t={T_max})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')

    plt.tight_layout()
    plt.show()

    # 等高线对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cont1 = ax1.contourf(X, Y, u0, levels=20, cmap='viridis')
    ax1.set_title('set')
    plt.colorbar(cont1, ax=ax1)

    cont2 = ax2.contourf(X, Y, u_final, levels=20, cmap='viridis')
    ax2.set_title('set')
    plt.colorbar(cont2, ax=ax2)

    plt.tight_layout()
    plt.show()

def task2_3d():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, sin, exp, pi
    from mpl_toolkits.mplot3d import Axes3D

    # ===================== 3D热方程求解 =====================
    # 参数设置
    k = 0.005  # 热扩散系数
    a = 1.0  # x方向长度
    b = 1.0  # y方向长度
    c = 1.0  # z方向长度
    T_max = 1.0  # 最大时间
    Nx = 20  # x方向离散点数
    Ny = 20  # y方向离散点数
    Nz = 20  # z方向离散点数
    N_terms = 2  # 级数项数（3D计算量较大）

    # 生成网格
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    z = np.linspace(0, c, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 初始条件
    u0 = np.sin(np.pi * X / a) * np.sin(np.pi * Y / b) * np.sin(np.pi * Z / c)

    # 最终时刻解
    u_final = np.zeros_like(X)
    expression_terms = []

    # 符号变量
    x_sym, y_sym, z_sym, t_sym = symbols('x y z t')

    # 分离变量法求解
    for m in range(1, N_terms + 1):
        for n in range(1, N_terms + 1):
            for p in range(1, N_terms + 1):
                # 特征值
                lambda_mnp = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2 + (p * np.pi / c) ** 2

                # 初始条件系数（仅m=n=p=1时有值）
                if m == 1 and n == 1 and p == 1:
                    Cmnp = 1.0
                else:
                    Cmnp = 0.0

                # 最终时刻解
                term = Cmnp * np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b) * np.sin(
                    p * np.pi * Z / c) * np.exp(
                    -k * lambda_mnp * T_max)
                u_final += term

                # 构造表达式
                if Cmnp != 0:
                    expr = f"{Cmnp}*sin({m}*pi*x/{a})*sin({n}*pi*y/{b})*sin({p}*pi*z/{c})*exp(-{k}*({m}²π²/{a}²+{n}²π²/{b}²+{p}²π²/{c}²)*t)"
                    expression_terms.append(expr)

    # ===================== 输出表达式 =====================
    print("\n" + "=" * 80)
    print("3D热方程解的表达式：")
    print("u(x,y,z,t) = " + " + ".join(expression_terms))
    print("=" * 80)

    # ===================== 可视化结果（切片展示） =====================
    fig = plt.figure(figsize=(15, 10))

    # 初始时刻切片（z=0.5）
    ax1 = fig.add_subplot(231, projection='3d')
    slice_z05_initial = u0[:, :, Nz // 2]
    X_slice, Y_slice = np.meshgrid(x, y)
    ax1.plot_surface(X_slice, Y_slice, slice_z05_initial.T, cmap='viridis')
    ax1.set_title('start (z=0.5)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # 最终时刻切片（z=0.5）
    ax2 = fig.add_subplot(232, projection='3d')
    slice_z05_final = u_final[:, :, Nz // 2]
    ax2.plot_surface(X_slice, Y_slice, slice_z05_final.T, cmap='viridis')
    ax2.set_title(f'final (z=0.5, t={T_max})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # 初始时刻切片（x=0.5）
    ax3 = fig.add_subplot(233, projection='3d')
    slice_x05_initial = u0[Nx // 2, :, :]
    Y_slice, Z_slice = np.meshgrid(y, z)
    ax3.plot_surface(Y_slice, Z_slice, slice_x05_initial.T, cmap='viridis')
    ax3.set_title('start (x=0.5)')
    ax3.set_xlabel('y')
    ax3.set_ylabel('z')

    # 最终时刻切片（x=0.5）
    ax4 = fig.add_subplot(234, projection='3d')
    slice_x05_final = u_final[Nx // 2, :, :]
    ax4.plot_surface(Y_slice, Z_slice, slice_x05_final.T, cmap='viridis')
    ax4.set_title(f'final (x=0.5, t={T_max})')
    ax4.set_xlabel('y')
    ax4.set_ylabel('z')

    # 初始时刻切片（y=0.5）
    ax5 = fig.add_subplot(235, projection='3d')
    slice_y05_initial = u0[:, Ny // 2, :]
    X_slice, Z_slice = np.meshgrid(x, z)
    ax5.plot_surface(X_slice, Z_slice, slice_y05_initial.T, cmap='viridis')
    ax5.set_title('final (y=0.5)')
    ax5.set_xlabel('x')
    ax5.set_ylabel('z')

    # 最终时刻切片（y=0.5）
    ax6 = fig.add_subplot(236, projection='3d')
    slice_y05_final = u_final[:, Ny // 2, :]
    ax6.plot_surface(X_slice, Z_slice, slice_y05_final.T, cmap='viridis')
    ax6.set_title(f'final (y=0.5, t={T_max})')
    ax6.set_xlabel('x')
    ax6.set_ylabel('z')

    plt.tight_layout()
    plt.show()

    # ===================== 中心点温度变化 =====================
    # 提取中心点温度随时间的变化（理论解）
    t_vals = np.linspace(0, T_max, 100)
    center_temp = np.exp(-k * ((np.pi / a) ** 2 + (np.pi / b) ** 2 + (np.pi / c) ** 2) * t_vals)

    plt.figure(figsize=(8, 4))
    plt.plot(t_vals, center_temp, 'b-', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('heat')
    plt.title('3D')
    plt.grid(True)
    plt.show()

def task3_1d():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, sin, cos, pi

    # ===================== 1. 问题定义与参数设置 =====================
    # 1D波动方程：u_tt = c² u_xx（描述两端固定弦的振动）
    # 空间域：x ∈ [0, L]
    # 时间域：t ∈ [0, T_max]

    # 物理参数
    c = 1.0  # 波速（m/s）
    L = 2.0  # 弦长（m）
    T_max = 4.0  # 最大模拟时间（s）（取2倍周期，便于观察振动特性）

    # 数值计算参数
    Nx = 100  # 空间离散点数
    N_terms = 5  # 傅里叶级数项数（用于表达式显示）

    # 生成空间网格
    x = np.linspace(0, L, Nx)

    # ===================== 2. 初始条件(ICs)与边界条件(BCs) =====================
    # --- 初始条件 (ICs) ---
    u0 = np.sin(np.pi * x / L)  # 初始位移：u(x, 0) = sin(πx/L)（基频振动模式）
    v0 = np.zeros_like(x)  # 初始速度：u_t(x, 0) = 0（静止释放）

    # --- 边界条件 (BCs)：齐次Dirichlet边界（两端固定） ---
    # BC1: u(0, t) = 0（左端固定）
    # BC2: u(L, t) = 0（右端固定）

    # ===================== 3. 分离变量法求解 =====================
    # 分离变量假设：u(x,t) = X(x)*T(t)
    # 代入波动方程得到两个常微分方程：
    # X'' + λX = 0 → X_n(x) = sin(nπx/L)（满足边界条件）
    # T'' + c²λT = 0 → T_n(t) = An cos(nπc t/L) + Bn sin(nπc t/L)
    # 通解：u(x,t) = Σ [sin(nπx/L) * (An cos(nπc t/L) + Bn sin(nπc t/L))]
    # 其中：
    # An = (2/L)∫₀ᴸ u0(x)sin(nπx/L)dx（初始位移系数）
    # Bn = (2/(nπc))∫₀ᴸ v0(x)sin(nπx/L)dx（初始速度系数）

    # 存储初始时刻和最终时刻的解
    u_initial = u0.copy()
    u_final = np.zeros_like(x)

    # 存储级数项表达式和系数
    expression_terms = []
    An_list = []
    Bn_list = []

    # 符号变量（用于表达式生成）
    x_sym, t_sym = symbols('x t')

    # 计算并叠加每一项级数
    for n in range(1, N_terms + 1):
        # 步骤1：计算An（初始位移系数）
        integrand_An = u0 * np.sin(n * np.pi * x / L)
        An = (2 / L) * np.trapz(integrand_An, x)
        An_list.append(round(An, 6))

        # 步骤2：计算Bn（初始速度系数，v0=0时Bn=0）
        integrand_Bn = v0 * np.sin(n * np.pi * x / L)
        Bn = (2 / (n * np.pi * c)) * np.trapz(integrand_Bn, x)
        Bn_list.append(round(Bn, 6))

        # 步骤3：计算最终时刻的级数项
        omega_n = n * np.pi * c / L  # 角频率
        T_n = An * np.cos(omega_n * T_max) + Bn * np.sin(omega_n * T_max)
        X_n = np.sin(n * np.pi * x / L)
        u_final += X_n * T_n

        # 步骤4：构造符号表达式
        if Bn == 0:
            term_expr = f"{An_list[-1]}*sin({n}*pi*x/{L})*cos({n}*pi*{c}*t/{L})"
        else:
            term_expr = f"{An_list[-1]}*sin({n}*pi*x/{L})*cos({n}*pi*{c}*t/{L}) + {Bn_list[-1]}*sin({n}*pi*x/{L})*sin({n}*pi*{c}*t/{L})"
        expression_terms.append(term_expr)

    # ===================== 4. 输出函数表达式 =====================
    print("=" * 80)
    print(f"1D波动方程近似解（前{N_terms}项）：")
    print("u(x,t) = " + " + ".join(expression_terms))
    print("\n系数：")
    for n in range(N_terms):
        print(f"An({n + 1}) = {An_list[n]}, Bn({n + 1}) = {Bn_list[n]}")
    print("=" * 80)

    # ===================== 5. 结果可视化 =====================
    plt.figure(figsize=(10, 6))

    # 绘制初始时刻和最终时刻的位移分布
    plt.plot(x, u_initial, label=f'Initial Time (t=0)', color='blue', linewidth=2)
    plt.plot(x, u_final, label=f'Final Time (t={T_max}s)', color='red', linewidth=2, linestyle='--')

    # 标注边界条件
    plt.scatter([0, L], [0, 0], color='black', s=50, label='BC: u(0,t)=u(L,t)=0')

    # 图表设置
    plt.xlabel('Position x (m)', fontsize=12)
    plt.ylabel('Displacement u(x,t)', fontsize=12)
    plt.title('1D Wave Equation Solution (Fixed-Fixed String Vibration)', fontsize=14)
    plt.xlim(0, L)
    plt.ylim(-1.2, 1.2)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

    # ===================== 6. 结果验证 =====================
    print(f"\n初始位移最大值：{np.max(u_initial):.4f}")
    print(f"最终位移最大值：{np.max(u_final):.4f}")
    print(f"边界条件验证：u(0, T_max) = {u_final[0]:.6f}, u(L, T_max) = {u_final[-1]:.6f}（理论应为0）")

def task3_2d():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, sin, cos, pi
    from mpl_toolkits.mplot3d import Axes3D

    # ===================== 1. 问题定义与参数设置 =====================
    # 2D波动方程：u_tt = c²(u_xx + u_yy)（描述矩形膜的振动）
    # 空间域：x∈[0,a], y∈[0,b]
    # 时间域：t∈[0, T_max]

    # 物理参数
    c = 1.0  # 波速（m/s）
    a = 1.0  # x方向膜长（m）
    b = 1.0  # y方向膜长（m）
    T_max = 2.0  # 最大模拟时间（s）

    # 数值计算参数
    Nx = 50  # x方向离散点数
    Ny = 50  # y方向离散点数
    N_terms = 3  # 级数项数（用于表达式显示）

    # 生成网格
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    X, Y = np.meshgrid(x, y)

    # ===================== 2. 初始条件与边界条件 =====================
    # --- 初始条件 ---
    u0 = np.sin(np.pi * X / a) * np.sin(np.pi * Y / b)  # 初始位移：基频模式
    v0 = np.zeros_like(X)  # 初始速度：静止释放

    # --- 边界条件（四边固定） ---
    # u(0,y,t)=u(a,y,t)=0, u(x,0,t)=u(x,b,t)=0

    # ===================== 3. 分离变量法求解 =====================
    # 通解：u(x,y,t) = ΣΣ [sin(mπx/a)sin(nπy/b)(Amn cos(ω_mn t) + Bmn sin(ω_mn t))]
    # ω_mn = c√((mπ/a)² + (nπ/b)²)（角频率）
    # Amn = (4/(ab))∫∫u0 sin(mπx/a)sin(nπy/b)dxdy（初始位移系数）
    # Bmn = (4/(abω_mn))∫∫v0 sin(mπx/a)sin(nπy/b)dxdy（初始速度系数）

    # 存储初始和最终时刻的解
    u_initial = u0.copy()
    u_final = np.zeros_like(X)

    # 符号变量
    x_sym, y_sym, t_sym = symbols('x y t')
    expression_terms = []

    # 计算级数解
    for m in range(1, N_terms + 1):
        for n in range(1, N_terms + 1):
            # 计算Amn和Bmn
            integrand_Amn = u0 * np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
            Amn = (4 / (a * b)) * np.trapz(np.trapz(integrand_Amn, x), y)

            integrand_Bmn = v0 * np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
            Bmn = (4 / (a * b * np.sqrt((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2))) * np.trapz(
                np.trapz(integrand_Bmn, x), y)

            # 计算角频率和最终时刻解
            omega_mn = c * np.sqrt((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2)
            T_mn = Amn * np.cos(omega_mn * T_max) + Bmn * np.sin(omega_mn * T_max)
            X_mn = np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)
            u_final += X_mn * T_mn

            # 构造表达式
            if Bmn == 0:
                term = f"{round(Amn, 6)}*sin({m}*pi*x/{a})*sin({n}*pi*y/{b})*cos({round(omega_mn, 4)}*t)"
            else:
                term = f"{round(Amn, 6)}*sin({m}*pi*x/{a})*sin({n}*pi*y/{b})*cos({round(omega_mn, 4)}*t) + {round(Bmn, 6)}*sin({m}*pi*x/{a})*sin({n}*pi*y/{b})*sin({round(omega_mn, 4)}*t)"
            expression_terms.append(term)

    # ===================== 4. 输出表达式 =====================
    print("=" * 80)

    print("u(x,y,t) = " + " + ".join(expression_terms))
    print("=" * 80)

    # ===================== 5. 结果可视化 =====================
    fig = plt.figure(figsize=(12, 5))

    # 初始时刻
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_initial, cmap='viridis')
    ax1.set_title('Initial Time (t=0)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('u(x,y,t)')

    # 最终时刻
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_final, cmap='viridis')
    ax2.set_title(f'Final Time (t={T_max}s)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_zlabel('u(x,y,t)')

    plt.tight_layout()
    plt.show()

    # ===================== 6. 结果验证 =====================
    print(f"\n初始位移最大值：{np.max(u_initial):.4f}")
    print(f"最终位移最大值：{np.max(u_final):.4f}")
    print(f"边界验证：u(0,0,T_max)={u_final[0, 0]:.6f}, u(a,b,T_max)={u_final[-1, -1]:.6f}（理论应为0）")

def task3_3d():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, sin, cos, pi
    from mpl_toolkits.mplot3d import Axes3D

    # ===================== 1. 问题定义与参数设置 =====================
    # 3D波动方程：u_tt = c²(u_xx + u_yy + u_zz)（描述立方体腔的振动）
    # 空间域：x∈[0,a], y∈[0,b], z∈[0,c]
    # 时间域：t∈[0, T_max]

    # 物理参数
    c = 1.0  # 波速（m/s）
    a = 1.0  # x方向长度（m）
    b = 1.0  # y方向长度（m）
    c_len = 1.0  # z方向长度（m）（避免与波速c重名）
    T_max = 1.0  # 最大模拟时间（s）

    # 数值计算参数
    Nx = 20  # x方向离散点数
    Ny = 20  # y方向离散点数
    Nz = 20  # z方向离散点数
    N_terms = 2  # 级数项数（3D计算量较大）

    # 生成网格
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    z = np.linspace(0, c_len, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # ===================== 2. 初始条件与边界条件 =====================
    # --- 初始条件 ---
    u0 = np.sin(np.pi * X / a) * np.sin(np.pi * Y / b) * np.sin(np.pi * Z / c_len)  # 初始位移
    v0 = np.zeros_like(X)  # 初始速度

    # --- 边界条件（六面固定） ---
    # u(0,y,z,t)=u(a,y,z,t)=0, u(x,0,z,t)=u(x,b,z,t)=0, u(x,y,0,t)=u(x,y,c_len,t)=0

    # ===================== 3. 分离变量法求解 =====================
    # 通解：u(x,y,z,t) = ΣΣΣ [sin(mπx/a)sin(nπy/b)sin(pπz/c_len)(Amnp cos(ω_mnp t) + Bmnp sin(ω_mnp t))]
    # ω_mnp = c√((mπ/a)² + (nπ/b)² + (pπ/c_len)²)（角频率）

    # 存储初始和最终时刻的解
    u_initial = u0.copy()
    u_final = np.zeros_like(X)

    # 符号变量
    x_sym, y_sym, z_sym, t_sym = symbols('x y z t')
    expression_terms = []

    # 计算级数解
    for m in range(1, N_terms + 1):
        for n in range(1, N_terms + 1):
            for p in range(1, N_terms + 1):
                # 计算Amnp和Bmnp
                integrand_Amnp = u0 * np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b) * np.sin(
                    p * np.pi * Z / c_len)
                Amnp = (8 / (a * b * c_len)) * np.trapz(np.trapz(np.trapz(integrand_Amnp, x), y), z)

                integrand_Bmnp = v0 * np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b) * np.sin(
                    p * np.pi * Z / c_len)
                omega_mnp = c * np.sqrt((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2 + (p * np.pi / c_len) ** 2)
                Bmnp = (8 / (a * b * c_len * omega_mnp)) * np.trapz(np.trapz(np.trapz(integrand_Bmnp, x), y), z)

                # 计算最终时刻解
                T_mnp = Amnp * np.cos(omega_mnp * T_max) + Bmnp * np.sin(omega_mnp * T_max)
                X_mnp = np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b) * np.sin(p * np.pi * Z / c_len)
                u_final += X_mnp * T_mnp

                # 构造表达式
                term = f"{round(Amnp, 6)}*sin({m}*pi*x/{a})*sin({n}*pi*y/{b})*sin({p}*pi*z/{c_len})*cos({round(omega_mnp, 4)}*t)"
                expression_terms.append(term)

    # ===================== 4. 输出表达式 =====================
    print("=" * 80)
    print(f"3D波动方程近似解（前{N_terms ** 3}项）：")
    print("u(x,y,z,t) = " + " + ".join(expression_terms))
    print("=" * 80)

    # ===================== 5. 结果可视化（切片展示） =====================
    fig = plt.figure(figsize=(15, 5))

    # z=0.5切片（初始时刻）
    ax1 = fig.add_subplot(131, projection='3d')
    slice_z05_initial = u_initial[:, :, Nz // 2]
    X_slice, Y_slice = np.meshgrid(x, y)
    ax1.plot_surface(X_slice, Y_slice, slice_z05_initial.T, cmap='viridis')
    ax1.set_title('Initial Time (z=0.5)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')

    # z=0.5切片（最终时刻）
    ax2 = fig.add_subplot(132, projection='3d')
    slice_z05_final = u_final[:, :, Nz // 2]
    ax2.plot_surface(X_slice, Y_slice, slice_z05_final.T, cmap='viridis')
    ax2.set_title(f'Final Time (z=0.5, t={T_max}s)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')

    # x=0.5切片（最终时刻）
    ax3 = fig.add_subplot(133, projection='3d')
    slice_x05_final = u_final[Nx // 2, :, :]
    Y_slice, Z_slice = np.meshgrid(y, z)
    ax3.plot_surface(Y_slice, Z_slice, slice_x05_final.T, cmap='viridis')
    ax3.set_title(f'Final Time (x=0.5, t={T_max}s)')
    ax3.set_xlabel('y (m)')
    ax3.set_ylabel('z (m)')

    plt.tight_layout()
    plt.show()

    # ===================== 6. 结果验证 =====================
    print(f"\n初始位移最大值：{np.max(u_initial):.4f}")
    print(f"最终位移最大值：{np.max(u_final):.4f}")
    print(f"边界验证：u(0,0,0,T_max)={u_final[0, 0, 0]:.6f}, u(a,b,c_len,T_max)={u_final[-1, -1, -1]:.6f}（理论应为0）")

def task4_1d():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, sin, sinh, pi  # 注意：避免SymPy变量与NumPy变量重名

    # ===================== 1. 问题定义与参数设置 =====================
    # 1D拉普拉斯方程：u_xx = 0（解为线性函数，描述1D稳态分布）
    # 空间域：x ∈ [0, L]

    # 几何参数
    L = 1.0  # 空间域长度（m）
    Nx = 50  # 空间离散点数

    # 生成数值网格（纯NumPy数组，避免与SymPy符号混淆）
    x = np.linspace(0, L, Nx)

    # ===================== 2. 边界条件 =====================
    # Dirichlet边界：u(0) = 0, u(L) = 1（两端固定值）

    # ===================== 3. 解析解求解 =====================
    # 1D拉普拉斯方程的解为线性函数：u(x) = (u(L)-u(0))/L * x + u(0)
    u_analytical = (1 - 0) / L * x + 0  # 即 u(x) = x

    # （若用分离变量法：解为u(x)=A+Bx，由边界条件确定A=0,B=1）

    # ===================== 4. 结果可视化 =====================
    plt.figure(figsize=(8, 4))
    plt.plot(x, u_analytical, color='blue', linewidth=2, label='Analytical Solution (u(x)=x)')

    # 标注边界条件
    plt.scatter([0, L], [0, 1], color='red', s=50, label='BC: u(0)=0, u(L)=1')

    # 图表设置
    plt.xlabel('Position x (m)')
    plt.ylabel('u(x)')
    plt.title('1D Laplace Equation Solution')
    plt.xlim(0, L)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===================== 5. 结果验证 =====================
    print(f"边界验证：u(0)={u_analytical[0]:.6f}, u(L)={u_analytical[-1]:.6f}（符合设定）")
    print(f"解的二阶导数：{np.mean(np.diff(np.diff(u_analytical)))}（理论应为0）")

def task4_2d():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, sin, sinh, pi as sym_pi  # 区分SymPy和NumPy的pi
    from mpl_toolkits.mplot3d import Axes3D

    # ===================== 1. 问题定义与参数设置 =====================
    # 2D Laplace Equation: ∇²u = u_xx + u_yy = 0
    # Domain: x∈[0,a], y∈[0,b]

    # Geometric parameters
    a = 1.0  # x-length (m)
    b = 1.0  # y-length (m)

    # Numerical parameters
    Nx = 50  # Discrete points in x
    Ny = 50  # Discrete points in y
    N_terms = 10  # Fourier series terms

    # Generate numerical grid (NumPy only)
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    X, Y = np.meshgrid(x, y)

    # ===================== 2. Boundary Conditions =====================
    # BC1: u(0,y) = 0, u(a,y) = 0 (left/right)
    # BC2: u(x,0) = 0, u(x,b) = np.sin(np.pi*x/a) (bottom/top)

    # ===================== 3. Solution via Separation of Variables =====================
    u = np.zeros_like(X)
    expression_terms = []

    # SymPy symbols (avoid conflict with numerical variables)
    x_sym, y_sym = symbols('x y')

    for n in range(1, N_terms + 1):
        # Calculate coefficient Cn
        integrand = np.sin(np.pi * x / a) * np.sin(n * np.pi * x / a)
        Cn = (2 / a) * np.trapz(integrand, x) / np.sinh(n * np.pi * b / a)

        # Calculate series term
        term = Cn * np.sin(n * np.pi * X / a) * np.sinh(n * np.pi * Y / a)
        u += term

        # Construct symbolic expression
        expr_term = f"{round(Cn, 6)}*sin({n}*pi*x/{a})*sinh({n}*pi*y/{a})"
        expression_terms.append(expr_term)

    # ===================== 4. Output Expression =====================
    print("=" * 80)
    print(f"2D Laplace Equation Solution (First 5 terms):")
    print("u(x,y) = " + " + ".join(expression_terms[:5]))
    print("=" * 80)

    # ===================== 5. Visualization =====================
    fig = plt.figure(figsize=(12, 5))

    # 3D Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, u, cmap='viridis', alpha=0.8)
    ax1.set_title('2D Laplace Solution (Surface)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('u(x,y)')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, u, levels=20, cmap='viridis')
    ax2.set_title('2D Laplace Solution (Contour)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    fig.colorbar(contour, ax=ax2)

    plt.tight_layout()
    plt.show()

    # ===================== 6. Validation =====================
    u_top = u[-1, :]
    u_top_theory = np.sin(np.pi * x / a)
    error = np.max(np.abs(u_top - u_top_theory))
    print(f"\nTop boundary error: {error:.6f} (smaller = more accurate)")
    print(f"Boundary check: u(0,y)={u[:, 0].max():.6f}, u(a,y)={u[:, -1].max():.6f}, u(x,0)={u[0, :].max():.6f}")

def task4_3d():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, sin, sinh, pi as sym_pi
    from mpl_toolkits.mplot3d import Axes3D

    # ===================== 1. Problem Definition =====================
    # 3D Laplace Equation: ∇²u = u_xx + u_yy + u_zz = 0
    # Domain: x∈[0,a], y∈[0,b], z∈[0,c]

    # Geometric parameters
    a = 1.0  # x-length (m)
    b = 1.0  # y-length (m)
    c = 1.0  # z-length (m)

    # Numerical parameters
    Nx = 20  # Discrete points in x
    Ny = 20  # Discrete points in y
    Nz = 20  # Discrete points in z
    N_terms = 2  # Series terms (reduce for computation efficiency)

    # Generate numerical grid
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    z = np.linspace(0, c, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # ===================== 2. Boundary Conditions =====================
    # BC1: u(0,y,z)=u(a,y,z)=0, u(x,0,z)=u(x,b,z)=0 (x/y boundaries)
    # BC2: u(x,y,0)=0, u(x,y,c)=np.sin(np.pi*x/a)*np.sin(np.pi*y/b) (z boundaries)

    # ===================== 3. Solution via Separation of Variables =====================
    u = np.zeros_like(X)
    expression_terms = []

    # SymPy symbols
    x_sym, y_sym, z_sym = symbols('x y z')

    for m in range(1, N_terms + 1):
        for n in range(1, N_terms + 1):
            # Calculate coefficient Cmn
            k_mn = np.sqrt((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2)
            integrand_2d = np.sin(np.pi * x / a) * np.sin(np.pi * y / b)[:, np.newaxis] * np.sin(
                m * np.pi * x / a) * np.sin(n * np.pi * y / b)[:, np.newaxis]
            Cmn = (4 / (a * b)) * np.trapz(np.trapz(integrand_2d, x, axis=1), y) / np.sinh(k_mn * c)

            # Calculate series term
            term = Cmn * np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b) * np.sinh(k_mn * Z)
            u += term

            # Construct symbolic expression
            expr_term = f"{round(Cmn, 6)}*sin({m}*pi*x/{a})*sin({n}*pi*y/{b})*sinh({round(k_mn, 4)}*z)"
            expression_terms.append(expr_term)

    # ===================== 4. Output Expression =====================
    print("\n" + "=" * 80)
    print(f"3D Laplace Equation Solution (First 3 terms):")
    print("u(x,y,z) = " + " + ".join(expression_terms[:3]))
    print("=" * 80)

    # ===================== 5. Visualization (Slices) =====================
    fig = plt.figure(figsize=(15, 5))

    # Slice at z=c (top boundary)
    ax1 = fig.add_subplot(131)
    slice_zc = u[:, :, -1]
    im1 = ax1.imshow(slice_zc.T, extent=[0, a, 0, b], origin='lower', cmap='viridis')
    ax1.set_title(f'Slice at z={c}')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    plt.colorbar(im1, ax=ax1)

    # Slice at y=b/2 (middle y)
    ax2 = fig.add_subplot(132)
    slice_yhalf = u[:, Ny // 2, :]
    im2 = ax2.imshow(slice_yhalf.T, extent=[0, a, 0, c], origin='lower', cmap='viridis')
    ax2.set_title(f'Slice at y={b / 2}')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('z (m)')
    plt.colorbar(im2, ax=ax2)

    # Slice at x=a/2 (middle x)
    ax3 = fig.add_subplot(133)
    slice_xhalf = u[Nx // 2, :, :]
    im3 = ax3.imshow(slice_xhalf.T, extent=[0, b, 0, c], origin='lower', cmap='viridis')
    ax3.set_title(f'Slice at x={a / 2}')
    ax3.set_xlabel('y (m)')
    ax3.set_ylabel('z (m)')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.show()

    # ===================== 6. Validation =====================
    u_zc = u[:, :, -1]
    u_zc_theory = np.sin(np.pi * X / a) * np.sin(np.pi * Y / b)
    error = np.max(np.abs(u_zc - u_zc_theory))
    print(f"\nTop boundary (z=c) error: {error:.6f}")
    print(
        f"Boundary checks: x=0/a: {u[0, :, :].max():.6f}/{u[-1, :, :].max():.6f}, y=0/b: {u[:, 0, :].max():.6f}/{u[:, -1, :].max():.6f}, z=0: {u[:, :, 0].max():.6f}")

task2_1d()
task2_2d()
task2_3d()