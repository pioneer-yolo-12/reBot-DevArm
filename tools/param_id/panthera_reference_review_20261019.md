# Panthera 参数辨识源码评审

日期：2026-10-19。固定版本：`fe77ca4e389b6343258d22bbc9086d04cc02622f`。
只读克隆至 `/tmp/panthera_review_20261019`；未运行其硬件程序，未打开本项目 CAN，未修改控制代码。

## 结论

值得借鉴的是完整动力学的可辨识性分析、激励设计和冻结参数后的闭环对照。它不是当前 RS00 腕部卡顿或位置周期纹波的现成修复方案。其辨识流水线主要使用模型合成力矩，不能替代本项目真实反馈采集和力矩标定。

## 辨识实现

1. [regressor.py](https://github.com/hrx2025lucky-lab/panthera-arm-control/blob/fe77ca4e389b6343258d22bbc9086d04cc02622f/panthera/identification/regressor.py)：Pinocchio 刚体回归矩阵后追加 `diag(qdd)`、`diag(qd)`、`diag(tanh(200*qd))`，对应转子等效惯量、黏性和库仑摩擦。关系为 `tau = Y_link*pi_link + Ia*qdd + Fv*qd + Fc*tanh(200*qd)`。六轴模型共 78 个参数；在随机状态上堆叠 Y，通过 SVD 提取可辨识子空间 P。仓库测试对此模型声明 rank=52；本项目必须依自己的 URDF、固定夹爪与可观测轴重新计算，不能复用该秩或 P。
2. [excitation.py](https://github.com/hrx2025lucky-lab/panthera-arm-control/blob/fe77ca4e389b6343258d22bbc9086d04cc02622f/panthera/identification/excitation.py)：有限傅里叶参考轨迹，解析目标速度和加速度；随机搜索系数，以 `cond(W@P)` 为目标，同时检查采样点的位置、速度、可选加速度约束。条件数直接从矩阵 SVD 计算，不构造正规方程。
3. [offline_ls.py](https://github.com/hrx2025lucky-lab/panthera-arm-control/blob/fe77ca4e389b6343258d22bbc9086d04cc02622f/panthera/identification/offline_ls.py)：在投影坐标中增量 QR，把力矩作为增广列一起分解；解基参数 beta，再构造一个完整空间代表 `pi_hat=P@beta`。
4. [pipeline.py](https://github.com/hrx2025lucky-lab/panthera-arm-control/blob/fe77ca4e389b6343258d22bbc9086d04cc02622f/panthera/identification/pipeline.py)：`collect()` 沿解析轨迹调用 RNEA 产生力矩，并可加高斯噪声；`identify()` 求解，`holdout_check()` 换另一条轨迹评价。这里不是实测力矩。仿真可用真值比较 beta；真机没有这个真值，应以独立预测、重复性和物理校验评价。

## 对本项目的借鉴顺序

| 优先级 | 建议 | 与现状的关系 |
|---|---|---|
| 立即 | 给现有单轴动态诊断补充参数相关性、奇异值谱、按整次往返重采样的系数稳定性报告 | 当前已有列缩放最小二乘、秩/条件数和独立运行验证；重点是解释惯性/Fv 对求导窗口敏感，不必替换求解器 |
| 立即 | 保留换速度、换起点及冻结模型验证 | 本项目的位置周期项是必要候选；对方只有速度摩擦项，不能吸收约 0.0449 rad 的角度纹波后称其为摩擦 |
| 条件满足后 | 建立 B601-RS 自己的刚体+转子惯量+速度摩擦回归器，先仿真检查 `Y*pi` 与逆动力学一致性及基参数秩 | 扩展当前固定其他轴的 J_eff 模型；尚不宜直接六轴联动 |
| 再后 | 针对本机约束优化低幅傅里叶轨迹，做独立轨迹预测 | 要先完成可靠反馈及力矩校验；解析导数是参考导数，不能冒充当前卡顿情况下的实测导数 |
| 最后 | 冻结参数做基线/补偿闭环对照 | 离线预测改善不能直接批准补偿；比较跟踪误差、峰值、饱和、转向、停止与温升 |

本机原始采集的内核时间戳、逐帧反馈、新鲜度、日志与运动线程隔离继续保留。所有真机动作仍必须从折叠过渡 HOME，并在退出时 HOME → 折叠零位 → 确认停稳 → 失能。

## 不能照搬的细节

- **没有位置周期模型。**`regressor()` 的摩擦列只与速度有关，没有位置正弦/余弦、电角度或齿槽补偿项；无法直接解释我们已观察到的 J6 纹波。
- **轨迹优化并未硬约束力矩。**`optimize()` 的 cost 只调用 `violation()` 和条件数，`tau_max` 只在后续 `evaluate(..., rnea_fn=...)` 中检查。有限采样和罚函数也不是连续轨迹可执行性保证。移植时应增加最终硬拒绝，并检查预测及实测力矩、碰撞和温升。
- **示例限值与推荐接口不一致。**`identify_demo.py` 手工指定速度 2 rad/s、未给加速度约束，未使用 `official_limits()`；不复制其运动数值。参见 [示例](https://github.com/hrx2025lucky-lab/panthera-arm-control/blob/fe77ca4e389b6343258d22bbc9086d04cc02622f/panthera/demos/identify_demo.py)。
- **零边界只保证速度。**`zero_boundary()` 将 a 每行去均值，使周期端点速度为零；一般 `q(0)!=q0`，且 `qdd(0)` 不为零，需要从实测姿态平滑接入，不能将其当作 HOME 过渡实现。
- **完整参数代表未必物理可实现。**最小范数的 `P@beta` 没有施加质量正值、惯量物理一致性等约束；不能直接据此覆盖 URDF，特别不能仅凭训练残差小就投入 CTC。

## 实际复核发现：QR 残差漏项

用纯 NumPy 直接加载对方 `offline_ls.py`，无需驱动或仿真：

```python
A = np.array([[1.,0.], [2.,0.], [3.,0.], [4.,0.]])
y = np.array([1.,2.,3.,5.])
ls = OfflineBaseLS(np.eye(2))
ls.add(A, y)
beta, info = ls.solve()
```

该秩亏样例报告相对残差约 `1.42e-16`，而直接计算 `norm(A@beta-y)/norm(y)` 得 `0.109388`，约 10.94%。`solve()` 在秩亏或正则分支仅计算 `norm(R_A@beta-R_b)`，遗漏增广 QR 下方的残差分量；应保留该分量并合并平方。满秩无正则分支已有该分量。这里只复核这一数值缺陷，没有宣称对方全部测试或性能结论失效，也未替其修改代码。

## 闭环结果的启示

对方 [控制实验说明](https://github.com/hrx2025lucky-lab/panthera-arm-control/blob/fe77ca4e389b6343258d22bbc9086d04cc02622f/docs/02_%E8%BF%90%E5%8A%A8%E6%8E%A7%E5%88%B6%E4%B8%8E%E5%AE%9E%E9%AA%8C.md) 报告：冻结同一合成数据拟合结果后，在指定仿真负载及频率下 PD+g 误差下降约 26%–30%，CTC 误差反而增加约 15%–18%。这是其报告结果，本轮没有重新运行完整仿真。对我们最直接的启示是：力矩预测准确性、参数物理含义和补偿后的控制效果需要分别验证。

本轮只新增评审文档，不开启摩擦或位置纹波补偿；现有未标定协议力矩仍不能当作已验证 N*m。
