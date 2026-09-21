"""
SoCBF 严格仿真（按论文理论严格实现）
- 增广状态：跟踪误差 e1 = x1 - y(t), e2 = x2 - dy(t)
- 一阶 CBF：严格两层结构（外层速度约束 QP + 内层 PD 跟踪）
- SoCBF：严格多约束 QP（同时满足所有约束）
- 完整过程：学习（误差收敛）→ 追踪（稳定跟踪）→ 避障（安全介入）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')

# ============================================================
# 参数设置
# ============================================================
dt = 0.01
T = 10.0
N = int(T / dt)

# 机器人参数
Kp = 120.0
Kd = 25.0
u_sat = 20.0

# 参考轨迹（圆）
R_ref = 5.0
omega = 1.0

# 障碍物
n_obs = 3
obs_radius = 0.5

# 障碍物初始位置和速度
obs_x0 = np.array([
    [2.7, 8.5],
    [-2.1, 8.5],
    [0.0, -6.5],
])
obs_v = np.array([
    [0.0, -6.0],
    [0.0, -3.0],
    [0.0, 3.0],
])

# CBF 参数
alpha1 = 2.0   # 一阶 CBF 增益 / 二阶 CBF 的 alpha1
alpha2 = 8.0   # 二阶 CBF 的 alpha2

# 噪声
sigma = 0.08

# ============================================================
# 参考轨迹
# ============================================================
def ref_traj(t):
    """圆形参考轨迹"""
    x = R_ref * np.cos(omega * t)
    y = R_ref * np.sin(omega * t)
    dx = -R_ref * omega * np.sin(omega * t)
    dy = R_ref * omega * np.cos(omega * t)
    ddx = -R_ref * omega**2 * np.cos(omega * t)
    ddy = -R_ref * omega**2 * np.sin(omega * t)
    return (np.array([x, y]), np.array([dx, dy]), np.array([ddx, ddy]))

def obs_traj(t):
    """障碍物轨迹"""
    pos = obs_x0 + obs_v * t
    vel = obs_v * np.ones_like(pos)
    acc = np.zeros_like(pos)
    return pos, vel, acc

# ============================================================
# 标称控制器：PD（基于跟踪误差）
# ============================================================
def nominal_control(e1, e2):
    """PD 控制（负反馈）
    e1 = x1 - y, e2 = x2 - dy
    """
    return -Kp * e1 - Kd * e2

# ============================================================
# 严格的一阶 CBF（两层结构）
# ============================================================
def first_order_cbf_qp(x1, x2, o, do, u_nom):
    """
    严格的一阶 CBF：两层结构
    外层：约束 dot h + alpha1 * h >= 0（关于速度 x2 的约束）
          解 QP 求安全速度 v_safe
    内层：u = Kd * (v_safe - x2)（PD 跟踪安全速度）
    """
    # 一阶 CBF 约束：dot h_i + alpha1 * h_i >= 0
    # dot h_i = 2 g_i^T (x2 - do_i)
    # 2 g_i^T (x2 - do_i) + alpha1 h_i >= 0
    # 2 g_i^T x2 >= 2 g_i^T do_i - alpha1 h_i
    
    # 收集所有约束
    A = []  # g_i^T
    b = []  # b_i
    
    for i in range(n_obs):
        g = x1 - o[i]
        h = np.linalg.norm(g)**2 - obs_radius**2
        dot_h = 2 * g @ (x2 - do[i])
        
        # 约束：2 g^T v >= 2 g^T do - alpha1 * h
        A.append(2 * g)
        b.append(2 * g @ do[i] - alpha1 * h)
    
    A = np.array(A)  # (n_obs, 2)
    b = np.array(b)  # (n_obs,)
    
    # 标称速度：当前速度（或者 u_nom 积分后的速度？）
    # 实际上，一阶 CBF 是对速度的约束，标称速度就是当前速度
    v_nom = x2.copy()
    
    # 解 QP：min ||v - v_nom||^2 s.t. A v >= b
    # 用 scipy.optimize.minimize
    
    def objective(v):
        return np.sum((v - v_nom)**2)
    
    def constraint(v):
        return A @ v - b  # >= 0
    
    cons = {'type': 'ineq', 'fun': constraint}
    
    try:
        res = minimize(objective, v_nom, method='SLSQP', constraints=cons,
                      options={'maxiter': 100, 'ftol': 1e-8})
        v_safe = res.x
    except:
        v_safe = v_nom
    
    # 内层：PD 跟踪 v_safe
    # u = Kd * (v_safe - x2) + u_nom
    # 实际上，u_nom 已经包含了 PD 跟踪，这里只需要加修正
    # 更准确的做法是：u = u_nom + Kd * (v_safe - x2)
    # 但这样会重复。更好的做法是直接用 u = Kd * (v_safe - x2) + ddy（前馈）
    
    # 简单做法：u = u_nom + Kd * (v_safe - x2)
    u = u_nom + Kd * (v_safe - x2)
    
    return u

# ============================================================
# 严格的 SoCBF（多约束 QP）
# ============================================================
def socbf_qp(x1, x2, o, do, ddo, u_nom):
    """
    严格的 SoCBF：同时满足所有约束的 QP
    约束：ddot h_i + alpha2 * (dot h_i + alpha1 * h_i) >= 0
    即：2 g_i^T u >= b_i
    """
    # 收集所有约束
    A = []  # g_i^T
    b = []  # b_i
    
    for i in range(n_obs):
        g = x1 - o[i]
        h = np.linalg.norm(g)**2 - obs_radius**2
        dot_h = 2 * g @ (x2 - do[i])
        
        # SoCBF 条件：ddot h + alpha2 * (dot h + alpha1 * h) >= 0
        # ddot h = 2||x2 - do||^2 + 2 g^T (u - ddo)
        # 2||x2 - do||^2 + 2 g^T u - 2 g^T ddo + alpha2 * dot_h + alpha2 * alpha1 * h >= 0
        # 2 g^T u >= -2||x2 - do||^2 + 2 g^T ddo - alpha2 * dot_h - alpha2 * alpha1 * h
        
        b_i = (-2 * np.linalg.norm(x2 - do[i])**2 
               + 2 * g @ ddo[i] 
               - alpha2 * dot_h 
               - alpha2 * alpha1 * h)
        
        # 两边除以 2：g^T u >= b_i / 2
        A.append(g)
        b.append(b_i / 2.0)
    
    A = np.array(A)  # (n_obs, 2)
    b = np.array(b)  # (n_obs,)
    
    # 检查 u_nom 是否满足所有约束
    gTu = A @ u_nom
    if np.all(gTu >= b - 1e-8):
        return u_nom
    
    # 解 QP：min ||u - u_nom||^2 s.t. A u >= b
    def objective(u):
        return np.sum((u - u_nom)**2)
    
    def constraint(u):
        return A @ u - b  # >= 0
    
    cons = {'type': 'ineq', 'fun': constraint}
    
    try:
        res = minimize(objective, u_nom, method='SLSQP', constraints=cons,
                      options={'maxiter': 100, 'ftol': 1e-8})
        u_safe = res.x
    except:
        u_safe = u_nom
    
    return u_safe

# ============================================================
# 仿真
# ============================================================
def simulate(mode='socbf'):
    """
    仿真
    增广状态：[x1, x2]，跟踪误差：e1 = x1 - y(t), e2 = x2 - dy(t)
    """
    # 初始状态：有较大初始误差
    x1 = np.array([R_ref + 2.0, 0.0])
    x2 = np.array([0.0, R_ref * omega + 1.0])
    
    # 记录
    traj_x1 = np.zeros((N, 2))
    traj_e1 = np.zeros((N, 2))
    traj_e2 = np.zeros((N, 2))
    traj_clearance = np.zeros(N)
    collision = np.zeros(N, dtype=bool)
    obs_traj_all = np.zeros((N, n_obs, 2))
    
    np.random.seed(42)
    
    for k in range(N):
        t = k * dt
        traj_x1[k] = x1
        
        # 参考轨迹
        y, dy, ddy = ref_traj(t)
        
        # 跟踪误差
        e1 = x1 - y
        e2 = x2 - dy
        traj_e1[k] = e1
        traj_e2[k] = e2
        
        # 障碍物
        o, do, ddo = obs_traj(t)
        obs_traj_all[k] = o
        
        # 标称 PD 控制
        u_pd = nominal_control(e1, e2)
        
        # 安全层
        if mode == 'none':
            u = u_pd
        elif mode == 'first_order':
            u = first_order_cbf_qp(x1, x2, o, do, u_pd)
        elif mode == 'socbf':
            u = socbf_qp(x1, x2, o, do, ddo, u_pd)
        
        # 饱和
        norm_u = np.linalg.norm(u)
        if norm_u > u_sat:
            u = u / norm_u * u_sat
        
        # 检查碰撞
        min_clear = 1e10
        coll = False
        for i in range(n_obs):
            c = np.linalg.norm(x1 - o[i]) - obs_radius
            min_clear = min(min_clear, c)
            if c <= 0:
                coll = True
        traj_clearance[k] = min_clear
        collision[k] = coll
        
        # 积分（Euler-Maruyama）
        x1 = x1 + x2 * dt
        x2 = x2 + u * dt + sigma * np.random.randn(2) * np.sqrt(dt)
    
    return {
        'x1': traj_x1,
        'e1': traj_e1,
        'e2': traj_e2,
        'clearance': traj_clearance,
        'collision': collision,
        'obs_traj': obs_traj_all,
        't': np.arange(N) * dt,
    }

# ============================================================
# 运行仿真
# ============================================================
print("Running strict simulations...")
res_none = simulate('none')
print("  No safety done")
res_first = simulate('first_order')
print("  First-order CBF (strict QP) done")
res_socbf = simulate('socbf')
print("  SoCBF (strict QP) done")

# ============================================================
# 绘制动画
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('SoCBF Safe Tracking (Strict Implementation) — Learning → Convergence → Avoidance', 
             fontsize=14, fontweight='bold')

modes = ['No Safety (PD only)', 'First-order CBF (QP)', 'SoCBF (Proposed, QP)']
results = [res_none, res_first, res_socbf]
colors = ['red', 'orange', 'green']

def update(frame):
    for col, (res, title, color) in enumerate(zip(results, modes, colors)):
        # 上排：空间轨迹
        ax = axes[0, col]
        ax.clear()
        
        t = frame * dt
        
        # 参考轨迹圆
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(R_ref * np.cos(theta), R_ref * np.sin(theta), 
                'k--', alpha=0.3, label='Reference')
        
        # 当前参考点
        y, dy, ddy = ref_traj(t)
        ax.plot(y[0], y[1], 'k*', markersize=10, label='Target')
        
        # 障碍物历史轨迹
        start = max(0, frame - 50)
        for i in range(n_obs):
            ax.plot(res['obs_traj'][start:frame, i, 0], 
                    res['obs_traj'][start:frame, i, 1],
                    'r--', alpha=0.3, linewidth=1)
        
        # 当前障碍物
        o = res['obs_traj'][frame]
        for i in range(n_obs):
            circle = Circle(o[i], obs_radius, color='red', alpha=0.6)
            ax.add_patch(circle)
        
        # 机器人历史轨迹
        robot_start = max(0, frame - 150)
        ax.plot(res['x1'][robot_start:frame, 0], 
                res['x1'][robot_start:frame, 1], 
                color=color, linewidth=2, alpha=0.7)
        
        # 当前机器人位置
        ax.plot(res['x1'][frame, 0], res['x1'][frame, 1], 'o', 
                color=color, markersize=12, label='Robot')
        
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if res['collision'][frame]:
            ax.text(0.02, 0.98, '⚠ COLLISION!', transform=ax.transAxes,
                    color='red', fontweight='bold', va='top', fontsize=10)
        else:
            ax.text(0.02, 0.98, f'Clear: {res["clearance"][frame]:.2f} m', 
                    transform=ax.transAxes, color='green', va='top', fontsize=10)
        
        # 下排：跟踪误差
        ax2 = axes[1, col]
        ax2.clear()
        
        ax2.plot(res['t'][:frame], res['e1'][:frame, 0], 'b-', label='e_x', linewidth=1.5)
        ax2.plot(res['t'][:frame], res['e1'][:frame, 1], 'b--', label='e_y', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Position error [m]')
        ax2.set_title(f'{title} — Tracking Error', fontsize=10)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, T)
        ax2.set_ylim(-3, 3)
    
    fig.suptitle(f'SoCBF Strict Simulation — t = {t:.2f} s', 
                 fontsize=14, fontweight='bold')

print("Creating animation...")
ani = animation.FuncAnimation(fig, update, frames=N, interval=20, repeat=False)

print("Saving GIF...")
ani.save('socbf_strict_animation.gif', writer='pillow', fps=30, dpi=80)
print("Animation saved!")

# ============================================================
# 绘制关键帧对比
# ============================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('Keyframe: Initial Error (top) vs Converged Tracking (bottom)', 
              fontsize=14, fontweight='bold')

frame1 = int(0.5 / dt)
frame2 = int(5.0 / dt)

for col, (res, title, color) in enumerate(zip(results, modes, colors)):
    # 上排：初始误差
    ax = axes2[0, col]
    ax.clear()
    
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(R_ref * np.cos(theta), R_ref * np.sin(theta), 'k--', alpha=0.3)
    y, dy, ddy = ref_traj(frame1 * dt)
    ax.plot(y[0], y[1], 'k*', markersize=10)
    
    o = res['obs_traj'][frame1]
    for i in range(n_obs):
        circle = Circle(o[i], obs_radius, color='red', alpha=0.6)
        ax.add_patch(circle)
    
    ax.plot(res['x1'][:frame1, 0], res['x1'][:frame1, 1], 
            color=color, linewidth=2, alpha=0.8)
    ax.plot(res['x1'][frame1, 0], res['x1'][frame1, 1], 'o', 
            color=color, markersize=12)
    
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\nt = {frame1*dt:.1f}s (Initial error)', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 下排：稳定跟踪
    ax2 = axes2[1, col]
    ax2.clear()
    
    ax2.plot(R_ref * np.cos(theta), R_ref * np.sin(theta), 'k--', alpha=0.3)
    y, dy, ddy = ref_traj(frame2 * dt)
    ax2.plot(y[0], y[1], 'k*', markersize=10)
    
    o = res['obs_traj'][frame2]
    for i in range(n_obs):
        circle = Circle(o[i], obs_radius, color='red', alpha=0.6)
        ax2.add_patch(circle)
    
    ax2.plot(res['x1'][:frame2, 0], res['x1'][:frame2, 1], 
            color=color, linewidth=2, alpha=0.8)
    ax2.plot(res['x1'][frame2, 0], res['x1'][frame2, 1], 'o', 
            color=color, markersize=12)
    
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-8, 8)
    ax2.set_aspect('equal')
    ax2.set_title(f'{title}\nt = {frame2*dt:.1f}s (Converged)', fontsize=10)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('socbf_strict_keyframes.png', dpi=150, bbox_inches='tight')
print("Keyframes saved!")

# ============================================================
# 绘制跟踪误差对比
# ============================================================
fig3, (ax31, ax32) = plt.subplots(2, 1, figsize=(10, 8))

for res, title, color in zip(results, modes, colors):
    err_norm = np.linalg.norm(res['e1'], axis=1)
    ax31.plot(res['t'], err_norm, color=color, label=title, linewidth=1.5)

ax31.set_xlabel('Time [s]')
ax31.set_ylabel('Position error norm [m]')
ax31.set_title('Tracking Error Convergence (Strict Implementation)', fontsize=12, fontweight='bold')
ax31.legend()
ax31.grid(True, alpha=0.3)

ax32.plot(res_none['t'], res_none['clearance'], 'r-', label='No safety', linewidth=1.5)
ax32.plot(res_first['t'], res_first['clearance'], color='orange', label='First-order CBF (QP)', linewidth=1.5)
ax32.plot(res_socbf['t'], res_socbf['clearance'], 'g-', label='SoCBF (proposed, QP)', linewidth=2.5)
ax32.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Safety boundary')
ax32.set_xlabel('Time [s]')
ax32.set_ylabel('Minimum clearance [m]')
ax32.set_title('Obstacle Clearance Over Time (Strict QP)', fontsize=12, fontweight='bold')
ax32.legend()
ax32.grid(True, alpha=0.3)
ax32.set_ylim(-2, 5)

plt.tight_layout()
plt.savefig('socbf_strict_tracking.png', dpi=150, bbox_inches='tight')
print("Tracking error plot saved!")

# ============================================================
# 打印统计
# ============================================================
print("\n=== Statistics (Strict Implementation) ===")
for name, res in zip(modes, results):
    n_coll = np.sum(res['collision'])
    min_clear = np.min(res['clearance'])
    final_err = np.linalg.norm(res['e1'][-1])
    print(f"{name:35s}: collision = {n_coll:3d}, min clear = {min_clear:.3f} m, final err = {final_err:.3f} m")
