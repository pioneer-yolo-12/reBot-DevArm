# 开源辨识仓库对照：对 B601-RS / J6 实验的适用性

评审日期：2026-10-16。静态阅读仓库 `windiff886/Robot-Parameter-Indentification-Simulation`，固定版本 `c57e5eb50faf27cce30f90381bdcc5323cc21ead`。未构建或运行外部实机程序，未发送机械臂命令；本次不修改本项目的控制、退出和参数部署规则。

## 结论

有可借鉴的辨识方法，但没有现成的 RS 电机匀速控制修复方案，也没有针对位置周期力矩纹波的实现。最有用的是“实际运动状态 + 刚体动力学 + 非线性摩擦”的联合辨识路线。它可以作为 Method D 之外的研究分支；不能把当前未通过匀速门限的数据直接送入原 Method D，也不能仅凭拟合误差小就部署补偿。

## 值得借鉴的实现

### 非线性速度摩擦

`src/identification/src/algorithms.cpp` 中的 `NonlinearFrictionLM` 实现了六参数速度函数：

```text
f(v) = g1 * [tanh(g2*v) - tanh(g3*v)] + g4*tanh(g5*v) + g6*v
```

它以多初值 Levenberg–Marquardt 拟合，并通过指数参数化保持摩擦参数为正。该平滑速度模型比单纯 Fc/Fv 能描述更丰富的低速形状；它不含位置 q，也不等于包含预滑动、记忆效应的完整静摩擦动力学模型。仓库的力矩预测使用 `W*beta - f(v)`，移植时必须先核对本项目电机侧/关节侧力矩、方向及阻力项符号，不能照抄减号。

源码：[算法实现](https://github.com/windiff886/Robot-Parameter-Indentification-Simulation/blob/c57e5eb50faf27cce30f90381bdcc5323cc21ead/src/identification/src/algorithms.cpp#L270)。

### 联合动力学辨识

算法使用刚体回归矩阵和转子等效惯量项，并对非线性摩擦路径排除重复的线性 damping 参数。这意味着可以在变速轨迹上估计参数，而不必假设惯性项处处为零。前提是实际 q、qd、qdd 和反馈力矩可靠、激励足够、模型与机构匹配。

傅里叶激励的解析 qd/qdd 是**目标轨迹导数**。实机跟踪存在明显波动时，不能用它们冒充实际速度和加速度。当前项目基于实际接收时间的局部位置拟合仍应保留，并评估估计误差。

源码：[观测矩阵及算法切换](https://github.com/windiff886/Robot-Parameter-Indentification-Simulation/blob/c57e5eb50faf27cce30f90381bdcc5323cc21ead/src/identification/src/main.cpp#L414)、[傅里叶轨迹](https://github.com/windiff886/Robot-Parameter-Indentification-Simulation/blob/c57e5eb50faf27cce30f90381bdcc5323cc21ead/src/force_node/src/trajectory/fourier_trajectory.cpp)。

## 不能直接照搬的部分

| 位置 | 代码事实 | 对本项目的影响 |
|---|---|---|
| C++ `ExperimentRecorder::record()` | CSV 的 tau 列写入 `command.torque[i]`，没有写 `state.effort[i]` | 不能据此声称记录到了实测驱动力矩；继续分别保存 tau_fb、tau_ff、主机 PD 代理量 |
| C++ 控制器 → Piper 后端 → SDK 适配器 | 主机已经计算 PD 力矩，又把非零 kp/kd 和该力矩送入 MIT 接口 | 按标准 MIT 前馈语义存在重复叠加 PD 的风险；需要核对 SDK，不能移植该控制链路 |
| Python `CsvRecorder` 与 C++ `DataLoader` | 六轴 Python CSV 带 tau_cmd，共 31 列；C++ 加载器只接受 25 或 19 列 | 两条入口并未做到数据格式完全互通，需要明确导出适配 |
| 实机采样及预处理 | SDK 适配器使用读取时刻；qdd 主要由速度差分获得，未实现本项目的逐帧内核时间与新鲜度校验 | 不能解决反馈缓存、时间错位及速度噪声；已有采集机制应保留 |
| `BridgeSession.STOP` / Python runner finally | 直接 disable/disconnect，没有先折叠并确认 | 与本项目用户要求冲突，不移植退出逻辑 |
| 模型与实机 SDK | 支持 Panda/Piper，硬件为 Piper SDK | 不能直接用于 B601-RS；必须使用本项目 URDF、坐标约定、RS 协议及力矩标定 |

对应源码：[C++ 记录器](https://github.com/windiff886/Robot-Parameter-Indentification-Simulation/blob/c57e5eb50faf27cce30f90381bdcc5323cc21ead/src/app/experiment_recorder.cpp#L45)、[控制器](https://github.com/windiff886/Robot-Parameter-Indentification-Simulation/blob/c57e5eb50faf27cce30f90381bdcc5323cc21ead/src/force_node/src/force_controller.cpp#L224)、[MIT 适配器](https://github.com/windiff886/Robot-Parameter-Indentification-Simulation/blob/c57e5eb50faf27cce30f90381bdcc5323cc21ead/src/piper_real/sdk_adapter.py)、[Python CSV](https://github.com/windiff886/Robot-Parameter-Indentification-Simulation/blob/c57e5eb50faf27cce30f90381bdcc5323cc21ead/src/piper_real/recorder.py)、[CSV 加载器](https://github.com/windiff886/Robot-Parameter-Indentification-Simulation/blob/c57e5eb50faf27cce30f90381bdcc5323cc21ead/src/identification/src/data_loader.cpp)、[退出实现](https://github.com/windiff886/Robot-Parameter-Indentification-Simulation/blob/c57e5eb50faf27cce30f90381bdcc5323cc21ead/src/piper_real/bridge.py#L48)。

## 针对当前 J6 的下一步

1. **先变速度验证周期特征。**现有两组对照均为 0.10 rad/s；尽管正反方向、不同 kd 的波峰按角度高度重复，仅凭同一速度的数据还不能排除时间周期振荡。固定 kp=25、kd=2.0、载荷和运动区间，设计 0.07 / 0.10 / 0.15 rad/s 的往返诊断，分别比较角度周期、时间频率和转向相位。本次只提出实验设计，不启动运动。
2. **建立离线动态诊断模型。**先从固定其他轴的 J6 单轴试验开始，研究重力、等效惯性、速度摩擦与可选位置周期项的分离。示意关系为 `tau_fb = g(q) + J_eff*qdd + f(qd) + r(q)`。其中 `r(q)=a*sin(k*q)+b*cos(k*q)` 是本项目基于实测现象提出的扩展，**不是上述仓库已有功能**。约 0.0449 rad 的周期只作初始线索，不固定为某个电机极数/减速比结论。
3. **先保证可辨识，再增加非线性参数。**当前只有单一目标速度，且速度与角度纹波耦合，不能据此可靠区分六参数摩擦和位置项。增加独立速度、重复实验后，再比较 Fc/Fv、非线性速度模型及加入位置项的模型，并检查参数相关性和留出预测。
4. **用独立运行验证。**按完整运行、速度或区间留出验证；不只随机拆同一周期的相邻采样。目标是验证跨运行/速度的预测，而不只是解释训练曲线。
5. **最后考虑补偿。**仍需完成力矩单位、比例及偏置标定。诊断模型不输出可部署补偿文件；确认来源并完成独立验证后，再做限幅、缓慢渐入的补偿实验。所有动作继续遵守 HOME → 折叠零位 → 稳定确认 → 失能。

完整动力学方法意味着“非匀速不必一概丢弃”，不意味着坏时间戳、缺失反馈、未标定力矩或模型缺项可以忽略。原 Method D 的低加速度/匀速门限保持不变。

## 后续三步实施结果

用户随后授权实施三步诊断。已新增 `friction_id/dynamic.py` 及 `analyze_friction.py dynamic-fit / dynamic-validate`，并完成 0.07 / 0.10 / 0.15 rad/s 的训练与独立验证实机运行，每组各速度两次往返。原始记录与完整结果见[多速度诊断报告](realrobot_data/j6_multispeed_20261016/summary.md)。

多速度下角度周期保持约 0.0449 rad，时间频率随速度变化；冻结模型在独立运行上的平衡 RMSE 从 0.137 降至 0.046（协议力矩尺度，未标定）。求导窗口敏感性检查支持位置周期项，但黏性及惯性系数尚不足以作可靠物理解释。结论是将位置项与速度摩擦分开处理；完成驱动/反馈排查及力矩标定前，不启用在线补偿。两次采集均确认折叠到位后失能。
