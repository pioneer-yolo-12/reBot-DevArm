# J1/J2/J4 后续验证结果

本轮在此前用户确认的各轴 HOME→正向 .55 rad 空间内完成四次新增运行，共七次往返。每次均折叠→HOME→实验→HOME→折叠确认→失能；无故障、无命令发送失败。J5/J6 不做独立动作，未加载摩擦或周期补偿，未改变生产增益默认值或质量门限。

## J2 固定 150/20 的速度筛查

| 速度 rad/s | 正向合格率 / 连续秒 | 反向合格率 / 连续秒 | 结果 |
|---|---|---|---|
| .06 | 92.3% / 2.458 | 82.2% / .779 | 仅正向通过 |
| .08 | 80.4% / .608 | 68.7% / .519 | 均未通过 |
| .12 | 77.9% / .731 | 68.6% / .389 | 均未通过 |

各段速度 CV 均≤15%。.06 反向虽然很接近 .8 s，仍按原门限判不通过；.08/.12 反向还存在合格比例不足。新增筛查尚未形成一档完整合格的正反配对，不扩展三次重复训练，不输出 Fc/Fv。较慢速度有帮助，但单纯降速并未保证全部通过。

数据：[J2 质量](realrobot_data/primary_joints_20261019/J2_speed_screen_quality.json)。

## J1 固定 80/10 的速度与起点对照

- 新增 .07/.12 rad/s，各一次往返。均未通过。.07 的 CV 15.78%/15.92%，低速相对波动加重；.12 的 CV 9.49%/8.20% 通过，但合格比例仅42.0%/54.6%，最长连续 .165/.191 s。
- 起点偏移 +.05 rad、行程 .50 rad，终点仍为 .55 rad。仍未通过。该偏移接近一个完整纹波周期，因此不能作为区分角相位和时间相位的主要证据。
- 再补起点偏移 +.025 rad、行程 .525 rad，终点 .55 rad。仍未通过，正反合格率48.6%/50.9%，最长 .230/.278 s。这个约半周期偏移提供更有效的相位区分。

数据：[速度筛查](realrobot_data/primary_joints_20261019/J1_speed_screen_quality.json)、[完整周期偏移](realrobot_data/primary_joints_20261019/J1_shifted_start_quality.json)、[半周期偏移](realrobot_data/primary_joints_20261019/J1_halfperiod_start_quality.json)。改变起点仍先到 HOME，再预定位；退出仍回 HOME 后折叠。

## 冻结模型跨运行检验

新增[离线脚本](realrobot_data/primary_joints_20261019/validate_spatial_diagnostic.py)，对比三个描述性模型：每方向的线性角度趋势；趋势+角度正弦；趋势+固定时间频率正弦。所有周期/频率、系数、相位及偏置只从训练日志确定，预测独立日志时不重新拟合。共用角区间只由两份日志配置确定；不依赖验证力矩选区间。核对关节、增益、HOME、载荷 ID、标定和模型哈希。

包含非稳态样本是为了诊断纹波，未把这些样本作为原 Method D 的合格样本；未做 Nm 重力相减，也不输出惯量/Fc/Fv。表内单位都是未经物理校验的协议力矩尺度，非已标定 Nm。整体 RMSE 为正反均衡汇总，各方向内按样本数计权；多速度采集并非速度档等权，逐档数据见 JSON。

| 训练→独立运行 | 趋势基线 RMSE | 角度模型 RMSE | 固定时间模型 RMSE |
|---|---|---|---|
| J1 .10→.07/.12 | .206 | .129 | .270 |
| J1 原起点→+.025 | .209 | .128 | .271 |
| J2 .10→.06/.08/.12 | .500 | .451 | .545 |
| J4 .10→.15 | .131 | .057 | .161 |

J1 训练选出的角周期约 .04990 rad；半周期起点实验正向的角度模型 RMSE .1246，固定时间模型 .3587。反向起点仍是同一个 .55 rad，时间模型 .1368 接近角度模型 .1313，这是本实验设计的预期区别，不能把两个方向都称作独立的起点相位检验。+.05 偏移下两模型接近，也符合接近完整周期偏移的混淆。

J4 训练角周期 .04482 rad；跨速度模型冻结后，角度模型仍显著优于趋势和时间模型。J1/J4 的位置相关分量已得到更强证据，但物理源头仍未确定。

J2 周期模型只带来约10%的 RMSE 改善，不能以同一正弦概括其全部力矩问题。另外在测量区间去除线性角趋势及 .04987 rad 正弦后，D=10 的力矩残差标准差约 .215/.221，D=20 约 .474/.455。提高 D 的运动平滑改善伴随更大的力矩残差；同步反馈速度与局部位置估计速度差的残差相关性较弱（|r|≤约.17），尚不足以把残差全部归因为 D 对反馈速度噪声的放大。此结果也不能直接解释为摩擦随 D 变大。

预测结果与图：

- [J1 跨速度](realrobot_data/primary_joints_20261019/J1_damping_to_J1_speed_screen/validation.json)
- [J1 半周期起点](realrobot_data/primary_joints_20261019/J1_damping_to_J1_halfperiod_start/validation.json)、[相位对照图](realrobot_data/primary_joints_20261019/J1_damping_to_J1_halfperiod_start/heldout.png)
- [J2 跨速度](realrobot_data/primary_joints_20261019/J2_damping_to_J2_speed_screen/validation.json)
- [J4 跨速度](realrobot_data/primary_joints_20261019/J4_pilot_to_J4_speed15/validation.json)、[预测图](realrobot_data/primary_joints_20261019/J4_pilot_to_J4_speed15/heldout.png)
- [反馈速度/力矩残差描述统计](realrobot_data/primary_joints_20261019/feedback_velocity_torque_diagnostic.json)

## 当前决策

停止本轮继续增加速度或重复调 D。J2 尚没有满足多速度配对要求的工况；J1/J4 应显式保留角度周期分支，不能把它吸收进速度摩擦。下一步优先离线评估含惯性、速度摩擦与位置周期项的可辨识性；J2/J3 必须处理随姿态变化的重力，使用同构型正反差分或经过物理标定后的一致单位重力模型，不能套常数偏置模型。诊断模型的预测提升不等于已获得可部署摩擦参数。

本轮最高温度56°C（保持 HOME 的 J3），运行前低于45°C；各轮之间冷却、只在失能后分析。结束的被动读回确认六轴 drive_state=0、fault=0，位置接近折叠零位，温度32～41°C。见[运行审计](realrobot_data/primary_joints_20261019/continuation_audit.json)、[结束状态](realrobot_data/primary_joints_20261019/continuation_final_state.json)。
