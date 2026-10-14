# reBot Arm B601-RS 参数辨识工具集

解决 RS 机械臂**低速运动不丝滑**问题的系统化工具。

## 问题诊断

低速运动卡顿（stick-slip）的根因通常有三层：

| 层面 | 现象 | 原因 |
|---|---|---|
| **摩擦力** | 启动时有明显的"咬死-释放"跳变 | 静摩擦/库仑摩擦未建模，控制器输出不足时静止，超过瞬间冲过头 |
| **惯性** | 加减速段跟踪差、过冲 | CAD 导出的惯性参数与实际有偏差（线束、装配体、材料密度差异） |
| **扰动** | 换向时抖动、带载时轨迹偏移 | 线束拖拽力、减速器 backlash、负载变化等未建模动态 |

## 辨识流程总览

```
┌─────────────────────────────────────────────────────────┐
│  1. 激励轨迹设计  →  2. 数据采集  →  3. 参数辨识         │
│                                                         │
│  4. URDF 更新     →  5. 摩擦补偿    →  6. 扰动观测器     │
└─────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 环境准备

```bash
# 核心依赖
pip install numpy scipy pin

# 可选：ROS 2 数据采集（如果有 ROS 环境）
# 数据采集依赖 ros2 topic 或你的电机驱动 SDK
```

### 第一步：激励轨迹设计

```python
from trajectory_design import generate_random_trajectories, trajectory_to_csv

# 生成 6 个关节的傅里叶级数激励轨迹
trajs = generate_random_trajectories(
    n_harmonics=5,
    base_freq=0.1,   # 0.1 Hz 基频 → 重点关注低速段
    seed=42,
)

# 导出 joint1 的轨迹到 CSV
csv_data = trajectory_to_csv("joint1", trajs["joint1"], duration=20.0)
with open("traj_joint1.csv", "w") as f:
    f.write(csv_data)

# 查看轨迹边界
for name, traj in trajs.items():
    ev = traj.evaluate_max()
    print(f"{name}: q=[{ev['q_min']:.2f}, {ev['q_max']:.2f}], "
          f"qd_max={ev['qd_max']:.2f}, qdd_max={ev['qdd_max']:.2f}")
```

### 第二步：数据采集

在真实机械臂上执行激励轨迹，记录每个时间步的：

- **q** — 关节角度（电机编码器）
- **q̇** — 关节角速度（编码器差分 或 陀螺仪）
- **q̈** — 关节角加速度（差分 或 加速度计）
- **τ_measured** — 关节力矩（电流反馈 × 转矩常数）

> **低速重点**：确保激励轨迹覆盖 0.01~0.5 rad/s 的速度区间，这是摩擦辨识的关键区域。

数据格式要求（CSV 或 NumPy）：

```
t,  q_j1, q_j2, q_j3, q_j4, q_j5, q_j6,  qd_j1, ..., tau_j1, ...
```

### 第三步：摩擦力辨识（推荐先做）

摩擦力辨识可以**独立于惯性参数**进行，是解决低速问题的首选步骤：

```python
from friction_model import (
    fit_coulomb_viscous, fit_stribeck, FrictionParams, PRIOR_FRICTION
)

# 方法 A：恒速实验（最简单）
#   对每个关节，在多个恒定速度下运行，记录平均力矩
#   （需要先做重力补偿，或只在水平关节上做）
fc, fv = fit_coulomb_viscous(
    velocity=[-2.0, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 2.0],
    torque=[-1.1, -0.8, -0.63, -0.57, -0.52, 0.48, 0.55, 0.62, 0.79, 1.05],
)
print(f"joint1: Coulomb={fc:.4f} N·m, Viscous={fv:.4f} (N·m)/(rad/s)")

# 方法 B：Stribeck 完整辨识（需要覆盖极低速 + 高速）
params = fit_stribeck(velocity_array, torque_array, v_stribeck_guess=0.01)
print(f"Coulomb={params.f_coulomb:.4f}, Viscous={params.f_viscous:.4f}, "
      f"Stiction={params.f_stiction:.4f}")
```

### 第四步：全身参数辨识

```python
from param_id_core import RSIdentification, JointTrajectory, inject_friction_into_urdf
import numpy as np

# 加载 URDF
ident = RSIdentification("Rebot_Arm_description/RS/urdf/ReBot_Arm_RS.urdf")

# 加载采集的数据
data = np.load("collected_trajectory.npz")  # 或从 ROS bag 导入
traj = JointTrajectory(
    t=data["t"],
    q=data["q"],
    qd=data["qd"],
    qdd=data["qdd"],
    tau_measured=data["tau"],
)

# 执行辨识
result = ident.identify(traj)
print(f"RMS residual: {result.rms_residual:.4f} N·m")
print(f"Condition number: {result.cond_number:.1f}")
for jname, (fc, fv, _) in result.friction.items():
    print(f"  {jname}: fc={fc:.4f}, fv={fv:.4f}")

# 将摩擦参数写入 URDF
inject_friction_into_urdf(
    "Rebot_Arm_description/RS/urdf/ReBot_Arm_RS.urdf",
    "Rebot_Arm_description/RS/urdf/ReBot_Arm_RS_with_friction.urdf",
    result.friction,
)
```

### 第五步：实时摩擦补偿 + 扰动观测器

```python
from disturbance_observer import (
    FrictionCompensator, DisturbanceObserver, CombinedObserverCompensator,
    DOBConfig, FrictionParams
)
from friction_model import RS_JOINTS

# 填入辨识结果
params_dict = {
    "joint1": FrictionParams(f_coulomb=0.52, f_viscous=0.38),
    "joint2": FrictionParams(f_coulomb=0.58, f_viscous=0.42),
    "joint3": FrictionParams(f_coulomb=0.47, f_viscous=0.35),
    "joint4": FrictionParams(f_coulomb=0.18, f_viscous=0.13),
    "joint5": FrictionParams(f_coulomb=0.19, f_viscous=0.14),
    "joint6": FrictionParams(f_coulomb=0.12, f_viscous=0.09),
}

# 各关节 DOB 配置（大关节惯量大，用更低截止频率）
dob_configs = {
    "joint1": DOBConfig(cutoff_freq=10.0, inertia_nominal=0.02),
    "joint2": DOBConfig(cutoff_freq=10.0, inertia_nominal=0.015),
    "joint3": DOBConfig(cutoff_freq=12.0, inertia_nominal=0.01),
    "joint4": DOBConfig(cutoff_freq=20.0, inertia_nominal=0.003),
    "joint5": DOBConfig(cutoff_freq=20.0, inertia_nominal=0.003),
    "joint6": DOBConfig(cutoff_freq=20.0, inertia_nominal=0.002),
}

controller = CombinedObserverCompensator(params_dict, dob_configs)

# 在每个控制周期调用：
# tau_cmd = controller.step(tau_ctrl_from_pd, qd_measured)
```

---

## 文件说明

| 文件 | 功能 |
|---|---|
| `friction_model.py` | 库仑+粘滞+Stribeck+LuGre 摩擦模型定义与拟合 |
| `trajectory_design.py` | 傅里叶级数激励轨迹设计（满足关节限位） |
| `param_id_core.py` | 基于 Pinocchio 的全身参数辨识（惯性+摩擦） |
| `disturbance_observer.py` | DOB + 摩擦前馈 + 组合控制器 |

---

## 辨识策略建议

### 优先级排序

**先做摩擦力辨识**（最快见效，80%的低速问题由摩擦引起）：

1. 对每个关节单独做恒速实验 → 拟合 Coulomb + viscous
2. 将拟合参数填入 `friction_model.py` 的 `PRIOR_FRICTION`
3. 在控制器中加入摩擦前馈
4. 测试低速跟踪是否改善

**再做全身辨识**（如果需要惯性补偿或前馈控制）：

1. 设计 6 关节同步激励轨迹
2. 采集完整数据集
3. 运行 `param_id_core.py` 辨识
4. 更新 URDF 中的 inertial 参数

**最后加扰动观测器**（对付残余不丝滑）：

1. 在摩擦补偿基础上叠加 DOB
2. DOB 截止频率从低开始（5-10 Hz），逐步提高
3. 观察 τ̂_dob 的波动，如果高频噪声大则降低截止频率

### 低速特别注意事项

- 激励轨迹基频 ≤ 0.1 Hz，确保有足够的低速采样点
- 速度过零区（|q̇| < 0.02 rad/s）数据谨慎使用：静摩擦导致 sign 函数跳变
- 建议对正/负速度分别拟合，验证摩擦对称性
- 如果关节使用谐波减速器，摩擦 toruqe 波动有周期性（与输入轴旋转同步）

---

## 预期效果

| 指标 | 改善前（典型） | 改善后（预期） |
|---|---|---|
| 低速跟踪抖动 | ±0.5° 以上 | ±0.1° 以内 |
| 速度过零死区 | 明显卡顿 | 平滑过渡 |
| 轨迹跟踪 RMS | 较大 | 降低 40-60% |
| 电机电流噪声 | 高频补偿震荡 | 前馈降低控制器负担 |

---

## 依赖

- Python ≥ 3.8
- numpy, scipy
- [pinocchio](https://github.com/stack-of-tasks/pinocchio)（全身辨识需要）
- 可选：ROS 2（数据采集）、matplotlib（可视化）
