# J6 配置核对与换起点测试记录

## 实际完成状态

- 已完成六轴位置、运行模式与 J6 配置只读核查，快照见 [原始读取记录](realrobot_data/j6_configuration_audit_20261016.json)。核查时六轴在折叠附近、驱动状态为未使能，J6 母线约 47.90 V。
- 已完成换起点采集和离线对比代码，离线测试共 71 passed、1 skipped；软件通过不能替代真实换起点实验。
- 换起点实验尚未完成：基准采集初始化时失去全部电机回包，尚未使能或发送运动命令。中止记录为 `realrobot_data/friction_20261016_180053_542b5fe6.npz`，其中反馈帧和运动命令均为 0。
- 随后 can0 仍 UP、1 Mbps，J1/J6 位置寄存器再次只读查询均超时。等待电机侧供电/CAN 连接状态确认；接口 UP 不代表电机应答正常。
- 用户确认暂时没有已知载荷和测量力臂，物理力矩标定未实施，见 [准备方案](torque_calibration_plan_20261016.md)。

## 配置发现与解释

项目配置将 J6 设为 RS00，这不是设备型号自动识别结果。厂家 [RS00 手册（固定版本）](https://github.com/RobStride/Product_Information/blob/384cc561e1ff079c22f68b19908526094430febd/Product%20Literature/CAN2.0/RS00/RS00User%20Manual260713.pdf) 给出 28 极（14 对极）、10:1 减速比，因此一个电周期对应输出端：

```text
2*pi / (14*10) = 0.044879895 rad = 2.5714286 deg
```

此前多速度实测角度周期约 0.04488 rad，联合拟合周期为 0.0447764 rad。接近一个电周期是排查电角度相关因素的线索，不能据此区分齿槽转矩、换相、电流环、编码器误差或其他成因。

J6 的 `0x702C` 齿槽补偿开关读为 0；`0x702D` 精确初始化开关读为 0，后者不是“校准完成”标志。旧调试地址 CAN_ID 返回 0，与实际响应节点 6 不符；因此包含 Kt=0 在内的旧调试寄存器零值不作为可信物理参数。固件版本未核实：文档版本请求使用通信类型 4，与停止命令共享类型，本轮未发送。

只读核查仅开启主动上报订阅，未发送清错、使能、失能、模式切换、零位、校准或参数持久化指令。厂家齿槽校准要求空载及重启，不在装配机械臂上直接启用或执行。

## 通信恢复后的实验

保持机械臂折叠、原有夹爪/载荷/线缆布置和唯一控制进程。先确认六轴位置可读、反馈新鲜且无故障，再依次执行以下两组（从仓库根目录）：

```bash
python3 tools/param_id/run_phase2_pointmatch.py --mode tracking --joint 6 --speeds .10 --repeats 2 --kp 25 --kd 2.0 --start-offset 0 --seed 20261018 --load-id live_tracking_load_unverified_20261016 --execute
python3 tools/param_id/run_phase2_pointmatch.py --mode tracking --joint 6 --speeds .10 --repeats 2 --kp 25 --kd 2.0 --start-offset .02 --seed 20261019 --load-id live_tracking_load_unverified_20261016 --execute
```

第二组从 HOME 平滑到 J6=0.02，再扫至 0.57 rad；不改编码器零点。每组完成后先经 HOME 回折叠零位，持续确认停稳后才失能。第一组未正常完成并回位时不启动第二组。

采完将下列 REFERENCE/SHIFTED 替换为两次真实输出文件，再离线分析：

```bash
python3 tools/param_id/analyze_friction.py changed-start tools/param_id/realrobot_data/REFERENCE.npz tools/param_id/realrobot_data/SHIFTED.npz --model tools/param_id/realrobot_data/j6_multispeed_20261016/train_final/dynamic_diagnostic_model.json --output tools/param_id/realrobot_data/j6_changed_start_20261016 --plot
```

对比正反方向的绝对角度对齐误差和相对起点对齐误差；只使用共同实测范围，不重新拟合相位、幅值或偏置。冻结既有动态模型作附加预测。结果保持诊断用途，现有未标定力矩不转成可部署摩擦参数。
