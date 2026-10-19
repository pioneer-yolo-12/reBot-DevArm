# 非对称加减速设计与 J2 单档试采

## 完成内容

1. 统一入口增加 `--ramp-down-time`，不指定时沿用原对称斜坡；正反向均使用相同的时间顺序。保存 `trajectory`（斜坡类型、加速/减速时长、逐速度持续时间及参考加速度峰值）与 `dynamic_range`。
2. 配对动态分支增加 `--motion-scope full`，选择加速/恒速/减速完整运动段。默认仍为 constant，原 Method D 不变。完整模式要求新轨迹元数据，禁止对旧日志静默猜测配置。
3. 新增 `preview_dynamic_excitation.py`，直接使用生产 Sweep 和实际位置局部回归算法，比较对称/非对称、恒速/完整范围的设计矩阵。只输出参考特征、JSON 和图，不生成伪实测力矩或原始采集 NPZ。
4. 固定 J2 150/20、.10 rad/s，实机执行两组互换斜坡的试采，每组一次往返。每组均折叠→HOME→实验→HOME→折叠确认→失能；无故障或发送失败。J5/J6 没有独立动作，未加载新补偿。

## 离线设计结果

J1/J2 候选速度均 .06/.09/.12 rad/s，原端点 HOME→+.55 rad。加减速配置为 .8/2.0 s 与 2.0/.8 s，两组都保持原行程。最高参考加速度为 .28125 rad/s²；单档试采 .10 下为 .234375 rad/s²。这些是轨迹参考值，不是实测加速度限幅或碰撞认证。

同位置半差设计矩阵包含平滑库仑、速度、加速度及一个角度正弦/余弦对。参考预览的诊断判据在实测前固定：列秩5，列归一化条件数<100，将加速度列对其他列投影后的剩余 RMS≥.01 rad/s²。该判据只衡量参考激励，不是实机运动/补偿验收。

- 对称完整轨迹：估计加速度半差约 .00016 rad/s²，主要是有限采样和导数估计误差，达不到独立激励幅值；只看数值秩会误判。
- 非对称但仅分析恒速区：半差约 10⁻¹⁴ rad/s²，仍无惯性激励。
- 两组非对称完整轨迹合并：加速度半差 RMS约 .01308 rad/s²，投影后仍约 .01308；条件数约7.47，参考设计通过。

因此关键不只是增加减速参数，还必须让分析真正读取完整运动段。

预览：[J2 JSON](realrobot_data/primary_joints_20261019/asymmetric_preview_final/J2/preview.json)、[J2 图](realrobot_data/primary_joints_20261019/asymmetric_preview_final/J2/excitation.png)、[J1 JSON](realrobot_data/primary_joints_20261019/asymmetric_preview_final/J1/preview.json)。`asymmetric_preview` 是1/2秒中间设计，最终以 `asymmetric_preview_final` 的 .8/2 秒配置为准。

## J2 两次试采

| 运行 | 加速/减速秒 | 完整配对覆盖率 | 最大位置跟踪误差 | 加速度半差 RMS 实测/规划 |
|---|---|---|---|---|
| A | .8/2.0 | 100%（265 bins） | .00605 rad | .09191/.01354 rad/s² |
| B | 2.0/.8 | 100%（265 bins） | .00629 rad | .09371/约.01357 rad/s² |

单组实际波动明显大于规划激励，实际/规划半差相关性仅 .152/.161。原 Method D 恒速段仍未通过，未被完整模式覆盖或放宽。

对同一个实际角度，将两组的加速度半差再相减：

- 规划变化 RMS .02711 rad/s²，实测变化 RMS .03593 rad/s²。
- 实测变化与规划变化相关系数 **.816**：非对称轨迹的变化已经体现在运动反馈中。
- 对应力矩半差的组间变化 RMS **.31934**（未校验协议力矩尺度），与规划变化的相关系数仅 **.035**。
- 两组实际速度半差的差值 RMS .00119 rad/s，平滑库仑特征差值 RMS .00142；互换斜坡基本保留了速度摩擦输入，同时改变加速度输入。

这支持“新轨迹提供了额外运动激励”，仍不支持“力矩响应已足以可靠辨识惯性”。单对数据、小信号与未知力矩尺度下，低相关性不能解释为机械臂没有惯性或力矩传感器已坏。

## 力矩通道的附加排查

按 CAN 通信类型2（操作回复）和24（主动上报）分别查看，去除共同角趋势、方向偏置及一个角周期后：

| 运行 | 类型2残差均值/标准差 | 类型24残差均值/标准差 |
|---|---|---|
| A | −.0080/.456 | +.0081/.465 |
| B | −.0035/.463 | +.0035/.488 |

两类各自都较波动，未见一个足以解释现象的简单帧类型均值偏置。挑选间隔≤2 ms的两类帧，时间差中位约1.1 ms，位置差 RMS约 .00022 rad，力矩差 RMS却约 .59/.64。这里只是主机接收时间接近，不是同一电机采样时刻；无法由此判断真实瞬时力矩变化、内部上报/滤波时序或估计噪声各占多少。不能直接宣布丢弃其中一类即可解决。

另做角度域高斯平滑敏感性检查，参考与力矩变化使用相同处理，未做参数拟合：平滑标准差 .004/.008/.016 rad时，力矩组间差 RMS降至 .127/.097/.071，相关系数仍只有 .078/.067/.038。平滑能降低波动，但没有恢复清晰的规划激励响应；不以挑选最有利滤波强度发布参数。

数据：[帧类型检查](realrobot_data/primary_joints_20261019/asymmetric_pilot/review/frame_type_diagnostic.json)、[平滑敏感性](realrobot_data/primary_joints_20261019/asymmetric_pilot/review/torque_smoothing_sensitivity.json)。这些检查是描述统计，不是对驱动采样链路的物理标定。

## 冻结模型的跨斜坡预测

只用 A 训练，系数/周期/相位全部冻结后预测 B；均使用完整运动配对，不重拟合验证数据：

| 模型 | A 训练力矩重建 RMSE | B 独立运行 RMSE |
|---|---|---|
| 速度摩擦+角趋势 | .375 | .404 |
| 再加等效惯性 | .365 | .391 |
| 再加位置周期项 | .338 | .363 |

位置项有改善但残差仍大；单速度、每组一次往返不足以检验 Fc/Fv/J_eff 的可靠性。报告明确标记速度覆盖不足、重复不足且 `compensation_allowed=false`。不将本轮拟合值作为最终惯量或摩擦补偿。

## 本轮决策与下一步

完成了非对称轨迹、配置记录、完整段分析及一次互换斜坡实机对照。先不将它扩成多轴多速度批量采集：当前最需要解决的是相对于设计激励过大的力矩通道波动，以及尚未核实的物理尺度。

下一步优先检查 J2 的力矩上报采样/带宽与时序，完成保持位置及重复同一路径的力矩重复性检查；比较与运动导数带宽一致的处理，并明确保留原始值。若需提高激励，应先评估参考速度/加速度及重力支撑余量，再做少量对照。已知质量/力臂目前不可用，力矩标定仍未完成；不把等待标定替换为虚构 Nm 参数。

## 数据、复现和检查

原始数据：

- [A：.8/2.0](realrobot_data/primary_joints_20261019/asymmetric_pilot/J2_up08_down20/friction_20261019_153501_7e0e6240.npz)
- [B：2.0/.8](realrobot_data/primary_joints_20261019/asymmetric_pilot/J2_up20_down08/friction_20261019_153633_e053ffd2.npz)
- [配对、退出与模型审计](realrobot_data/primary_joints_20261019/asymmetric_pilot/review/summary.json)、[互换斜坡对比图](realrobot_data/primary_joints_20261019/asymmetric_pilot/review/mirror_comparison.png)
- [结束被动读回](realrobot_data/primary_joints_20261019/asymmetric_pilot/final_state.json)：六轴 drive_state=0、fault=0，接近折叠零位，温度32～34°C。两次运行最高温度为保持轴 J3 的47/48°C，运行前低于45°C。

完整分析示例（从 tools/param_id 执行；输出目录必须不存在已有模型文件）：

```bash
python3 analyze_friction.py paired-dynamic-fit realrobot_data/primary_joints_20261019/asymmetric_pilot/J2_up08_down20/friction_20261019_153501_7e0e6240.npz --motion-scope full --output /tmp/j2_asymmetric_fit
python3 analyze_friction.py paired-dynamic-validate realrobot_data/primary_joints_20261019/asymmetric_pilot/J2_up20_down08/friction_20261019_153633_e053ffd2.npz --model /tmp/j2_asymmetric_fit/paired_dynamic_model.json --output /tmp/j2_asymmetric_validation
```

测试覆盖正反轨迹连续/有界、默认兼容与元数据、无效斜坡参数、仅完整非对称轨迹能提供独立激励、模拟动态系数恢复、旧元数据拒绝及原回位/失能检查。`tools/param_id/tests`：93 passed，1 skipped。既有模型因实现哈希变化需重算后才能用新实现验证，历史报告未重写成新结果。
