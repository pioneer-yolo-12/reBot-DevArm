# B601-RS 六轴摩擦辨识与补偿

当前主线优先 J1～J4，按用户要求暂缓 J6 专项排查。J5/J6 在其他轴实验中继续保持姿态，不单独辨识或加载摩擦补偿。逐轴运动验收、训练与独立运行状态见 [四轴工作记录](primary_joints_plan_20261019.md)。摩擦辨识和重力模型校验分开进行；未标定协议力矩不得直接形成可部署 N*m 参数。

J2 反馈力矩链路可用 `python3 analyze_friction.py torque-audit RAW.npz --output /tmp/torque_audit --plot` 离线检查：原始载荷重算、主机接收/读取时间、分帧类型频谱及保持段的冻结延迟关联。它不估计真实电机采样延迟或修改时间戳。采集入口的 `--hold-time 3` 可延长 HOME/每次往返端点的诊断保持；默认仍为 .5 s，启动及折叠后失能流程不变。详见[力矩通道与重复性报告](torque_chain_review_20261019.md)。

## 同位置配对动态诊断（J1～J4）

新增 `paired-dynamic-fit` / `paired-dynamic-validate`，全程离线，无需 Motorbridge、CAN 或 Pinocchio。用于非匀速日志的模型诊断，不改变原 Method D 门限。正反向插值至相同实际角度，其他轴位置差要求≤.003 rad；保留完整时间、反馈新鲜度、局部拟合与故障检查。

```bash
# 以下命令从 tools/param_id 执行；可合并多个相同增益/载荷/模型的训练运行。
python3 analyze_friction.py paired-dynamic-fit realrobot_data/TRAIN.npz --output /tmp/paired_train
python3 analyze_friction.py paired-dynamic-validate realrobot_data/HELDOUT.npz --model /tmp/paired_train/paired_dynamic_model.json --output /tmp/paired_validation
```

力矩半差拟合 `Fc*Δtanh(v/.02)/2 + Fv*Δv/2 + J_eff*Δa/2 + 可选方向奇位置项`，力矩半和使用同一运动系数，另加三次角度趋势和可选方向偶位置项。半差尽量消去共同重力；半和趋势只是描述量，不是重力标定。未减去任何 Nm 制模型重力。默认仅取恒速指令区间；新日志可用 `paired-dynamic-fit --motion-scope full` 纳入完整运动段。此模式要求元数据含 `trajectory` 和 `dynamic_range`，不猜测旧日志的斜坡参数；验证沿用训练的模式。

非对称轨迹使用 `--ramp-time .8 --ramp-down-time 2`；正反两个方向均按相同的先加速、后减速时长执行，互换时长构成另一组独立运行。默认减速时长仍等于加速时长，原实验行为不变。NPZ 保存加减速时长、逐档持续时间和参考加速度峰值，启动/退出回位流程不变。可用 `python3 preview_dynamic_excitation.py --joint 2 --output /tmp/j2_excitation_preview` 生成无硬件预览；详见[非对称激励试采报告](asymmetric_excitation_review_20261019.md)。

输出含三种嵌套模型、条件数、负系数、按整次往返删除的系数稳定性及独立预测；空间周期只从训练数据选择。验证冻结全部系数和周期，校验源码/模型完整性、原始数据独立性及实验条件，拒绝训练速度范围外的预测。所有模型均 `compensation_allowed=false`，不能加载成 `validated` 参数；bin 数不是独立试验数。详见[四轴配对动态分析](primary_joints_dynamic_review_20261019.md)。

## 多速度周期诊断与离线动态分支

以下是此前 J6 的单轴直接回归分支。未标定力矩时其中的常数偏置只能描述近似恒定重力；J2/J3 随姿态变化的重力不能直接用此偏置替代，当前四轴优先使用上面的配对分支。

当真实速度存在周期波动时，原 Method D 的匀速门限保持不变；额外使用 `dynamic-fit` / `dynamic-validate` 分析非匀速样本。该分支是固定其他关节时的单轴等效模型，不是完整六轴惯性辨识，不输出可部署补偿参数。

固定增益、载荷和行程，分别采集训练组与独立验证组（以下命令从本目录执行）。每组完成后均须确认折叠到位才失能：

```bash
python3 run_phase2_pointmatch.py --mode tracking --joint 6 --speeds .07 .10 .15 --repeats 2 --kp 25 --kd 2.0 --seed 20261016 --load-id YOUR_FIXED_LOAD --execute
python3 run_phase2_pointmatch.py --mode tracking --joint 6 --speeds .07 .10 .15 --repeats 2 --kp 25 --kd 2.0 --seed 20261017 --load-id YOUR_FIXED_LOAD --execute
python3 analyze_friction.py dynamic-fit realrobot_data/TRAIN.npz --output /tmp/j6_dynamic_train
python3 analyze_friction.py dynamic-validate realrobot_data/HELDOUT.npz --model /tmp/j6_dynamic_train/dynamic_diagnostic_model.json --output /tmp/j6_dynamic_validation --plot
```

训练端只使用训练组搜索位置周期和时间频率，同时拟合三种候选：

- baseline：偏置 + 等效惯性 × 实测加速度 + 平滑库仑项 + 黏性项。
- spatial：baseline + `a*sin(2*pi*q/period) + b*cos(2*pi*q/period)`。
- temporal：baseline + 固定时间频率正弦/余弦；正反向分别有相位系数，但各速度、重复之间共享。

导数来自实际内核接收时间及局部位置回归，包含加减速阶段；导数窗口内的坏时间戳、陈旧反馈、缺轴、驱动故障及失败命令仍会被拒绝。各运动段等权，避免低速段或双路上报帧数量更多而支配结果。每个速度/方向/重复分别给出角度周期和时间频率诊断。

验证端要求不同 run ID、不同文件和原始事件指纹，并匹配关节、模型、载荷、标定、所有增益、采样频率、速度覆盖及温度范围。系数、偏置、频率、相位、求导窗口均不重新拟合；实现源码哈希变化后必须重新训练。支持位置项的预设诊断门限为：整体误差比 baseline 至少下降 20%，每个速度至少下降 10%，并至少优于固定时间模型 20%。这些门限只判断独立预测支持，不认证补偿安全或证明具体机械故障。

输出包括 `dynamic_diagnostic_model.json`、`dynamic_validation.json`、逐点 `validation_predictions.npz` 和 `dynamic_validation.png`。NPZ 保存惯性、速度摩擦、位置周期、时间周期、偏置及实际应用的重力分项。重力模型的 N*m 参考量单独保存为 `gravity_reference_nm`；**力矩未标定时不从协议值中减去 N*m 重力，也不将拟合惯量标为 kg·m²**。此时偏置可吸收近似常量的重力及零偏，所有系数只能作协议尺度诊断。

模型固定标记 `status=diagnostic_candidate`、`compensation_allowed=false`，不能由现有补偿入口当作 `validated` 摩擦参数加载。负惯性/摩擦系数、病态条件数和求导窗口敏感性需要单独审查；拟合误差小不能替代物理标定与干预实验。已有实机结果见 [2026-10-16 多速度报告](realrobot_data/j6_multispeed_20261016/summary.md)。

### 换起点诊断

固定 J6 增益、速度及载荷，分别从 0 和 0.02 rad 开始往返，检查纹波是否仍出现在相同的绝对关节角度。`--start-offset` 只改变实验起点，不改变编码器零点或 HOME；启动仍从折叠位过渡至 HOME，再平滑预定位。两组均执行 HOME → 折叠零位 → 确认 → 失能。

```bash
python3 run_phase2_pointmatch.py --mode tracking --joint 6 --speeds .10 --repeats 2 --kp 25 --kd 2.0 --start-offset 0 --seed 20261018 --load-id YOUR_FIXED_LOAD --execute
python3 run_phase2_pointmatch.py --mode tracking --joint 6 --speeds .10 --repeats 2 --kp 25 --kd 2.0 --start-offset .02 --seed 20261019 --load-id YOUR_FIXED_LOAD --execute
python3 analyze_friction.py changed-start realrobot_data/REFERENCE.npz realrobot_data/SHIFTED.npz --model /tmp/j6_dynamic_train/dynamic_diagnostic_model.json --output /tmp/j6_changed_start --plot
```

分析分别比较相同绝对角度和相对起点角度的力矩曲线，仅使用两种比较都覆盖的共同区间，不搜索相位、偏置或幅值；另用既有动态模型冻结预测。要求两组完整、独立且增益、载荷、模型、速度、行程等条件匹配。结果只用于判断位置相关性，不直接认证齿槽转矩成因或允许补偿。

2026-10-16 的只读配置见 [J6 配置与换起点记录](j6_configuration_and_shift_report_20261016.md)。已知载荷暂不可用，后续物理校验见 [力矩标定方案](torque_calibration_plan_20261016.md)，不得将方案当作已完成标定。

当前流程：**原始采集 → 单轴实际匀速验收 → Method D 主辨识、C/E 对照 → 独立运行验证 → 分轴补偿验证**。J4/J5/J6 尚需实机调参；软件检查通过不代表实机恒速或力矩标定已经完成。

本工具仅支持 RS 私有 CAN 协议、电机 ID 1–6、host ID 0xFD、RS-06×3 + RS-00×3，位置单位 rad。夹爪不参与本轮辨识，需固定夹爪、载荷和线缆布置。运行期间只能有一个运动命令发送程序。

## 安装与文件

Python 3.10+，离线部分依赖 numpy、scipy；图表另需 matplotlib；测试需 pytest。实机还需 motorbridge、本机 reBotArm_control_py 及其 Pinocchio 依赖。外部控制库可通过 `REBOT_CONTROL_ROOT` 指定，默认搜索本仓库相邻的 `reBotArm_control_py-1`/`reBotArm_control_py`。

```bash
cd tools/param_id
python3 -m pytest -q tests
```

仓库根目录也可执行 `python3 -m pytest -q`。CI 仅安装 numpy/scipy/pytest 跑离线测试，不要求 ROS、Motorbridge 或 CAN；MuJoCo 实验脚本不参与自动测试发现。

| 文件 | 用途 |
|---|---|
| `run_phase2_pointmatch.py` | 统一采集入口；tracking 调试 / identify 多速度重复辨识 |
| `run_phase2_gravity_ff.py` | 上述入口的兼容别名，删除旧的重复采集实现 |
| `analyze_friction.py` | 原始数据回放、质量检查、D/C/E 拟合、独立验证、日志恢复 |
| `friction_id/hardware.py` | 独立只读 SocketCAN 接收器和实机生命周期 |
| `friction_id/recording.py` | JSONL 日志、原子 NPZ 检查点和恢复 |
| `friction_id/analysis.py` | 实测运动门限、位置分箱配对、拟合和验证 |
| `gravity_friction_compensation.py` | 新反馈速度估计、参数有效范围和前馈限幅 |
| `11_gravity_friction_compensation.py` | 第四阶段入口，默认重力补偿、摩擦比例 0 |
| `test_phase2_sim.py` | MuJoCo 方法实验；运动原语复用 Sweep，修复负向减速位置跳变 |

## 第一阶段：采集链路

接收器使用独立的只读 SocketCAN socket，内核会向匹配的多个 socket 分发副本，不会读取 Motorbridge 自己的接收队列。项目不修改已安装的 SDK。参考 [SocketCAN 内核文档](https://docs.kernel.org/networking/can.html)及[时间戳文档](https://docs.kernel.org/networking/timestamping.html)。

每个接收帧记录原始 CAN ID、8 字节数据、内核 `SO_TIMESTAMPNS` 时间、主机读取时间、到 monotonic 的映射误差以及本机每轴接收序号。**内核接收时间不是电机内部采样时间；接收序号不是设备包序号，不能凭它证明总线无丢帧。**检测到本接收器队列溢出、时间映射跳变、陈旧/不完整反馈、驱动故障或命令发送失败会中止。

命令按每轴实际 `send_mit` 开始/结束时间记录，包括 q_ref、qd_ref、kp、kd、tau_ff、阶段、往返方向、速度档和重复编号。`tau_pd_host` 用对应接收时刻之前已发完的命令计算，只作主机侧代理量；未测得设备执行延迟时不宣称它等于内部 PD。

运动期间，控制和接收线程只向有界队列提交原始事件；独立 IO 线程写 append-only JSONL，并在磁盘响应正常时约每 0.5 秒执行 flush/fsync。磁盘慢不会让控制线程等待日志锁，队列满或写入失败则显式中止，不能静默丢数据。**运动及折叠回位期间不转换/压缩 NPZ**：Python 后台转换也会争用 GIL，使控制和接收同时停顿；`request_checkpoint()` 只记录延后导出标记。停止控制和接收后，才汇总完整 NPZ 和最终状态；异常杀进程留下的 JSONL 可用 recover 恢复。使能前读取全部位置并拦截无效值；初始化、HOME、中断和采集异常统一清理。HOME 固定为 `[0,1.2,1,0,0,0]`。

控制周期迟到超过 50 ms 仍会中止，不能通过放宽该门限掩盖写盘阻塞。回位超时诊断同时记录迟到毫秒数和发生阶段。

**所有真机入口退出时，必须先回到折叠零位 `[0,0,0,0,0,0]`，确认后才失能。**正常完成、Ctrl+C 和可恢复异常统一先平滑回 HOME，再平滑回折叠零位；已在折叠附近则直接收敛到零位。五次轨迹峰值速度不超过 0.3 rad/s，回位期间保留重力前馈。最后要求六轴位置误差均不超过 0.02 rad，至少 0.5 秒位置窗口内各轴变化不超过 0.002 rad、拟合速度不超过 0.02 rad/s。实机静止时驱动器瞬时速度存在约 0.2 rad/s 噪声，不能用瞬时速度近零来判断停稳；该速度仍用于 1 rad/s 越限保护。反馈无效、回位超时或跟踪越限时阻止失能，记录失败，仅关闭通信连接并报告需现场处理；关闭连接不保证驱动器继续保持，硬件看门狗行为取决于驱动器配置。启动要求处于折叠附近，避免初始化的清错/模式切换在展开状态隐含失能。J2/J3 零位附近允许最多 0.02 rad 的负读数偏差，仅用于启动向内恢复和折叠回位。

接收过滤器同时接收操作应答（通信类型 2）和本机固件的周期主动反馈（类型 24），两者原始 CAN ID 均随帧保存；不再漏掉周期主动上报。

从进程被终止后留下的日志恢复：

```bash
python3 analyze_friction.py recover realrobot_data/friction_XXX.jsonl --output /tmp/recovered.npz
```

恢复数据标为 recovered，不能成为可部署参数。最后一条被截断的 JSON 行允许丢弃，日志中间损坏会报错。

## 第二阶段：J1～J4 单轴运动质量验收

所有入口默认预览，不打开 CAN；`--execute` 才连接和运动。`--dry-run` 始终覆盖执行选项。

```bash
python3 run_phase2_pointmatch.py --joint 6 --mode tracking --speeds 0.10 --dry-run
# 确认支撑、固定载荷/夹爪/线缆和工作区后，在实机上执行：
python3 run_phase2_pointmatch.py --joint 6 --mode tracking --speeds 0.10 --load-id empty_fixed_gripper_v1 --execute
```

每次只改一个增益。例如 `--kp 25` 或 `--kd 1.5` 仅影响被调试轴，其余轴使用已有辨识增益。工具不自动提高增益；增益上限为原辨识基线的两倍，仍不是经过实机认证的安全规格。

原驱动腕部 PD 对照需显式加 `--gain-policy driver-reference`，仅允许 tracking 模式的 J4/J5/J6；只将所选轴的覆盖上限设为已核查原驱动值（J4=50/5、J5=50/4、J6=50/4），不会自动修改实际增益或其他轴。默认策略保持不变。策略和覆盖上限写入原始日志；原驱动值只是实验对照来源，不代表已认证稳定性。比如 J6 三组依次指定 `--kp 25 --kd 2`、`--kp 25 --kd 4`、`--kp 50 --kd 4`，共同保持 0.10 rad/s、100 Hz、相同行程及载荷；每组先做一次往返并确认回位，再判断是否进入下一组。所有力矩、速度、跟踪误差和折叠退出保护继续有效。

已完成的三组实机对照及 C/A 独立复测见 [2026-10-19 J6 PD 报告](realrobot_data/j6_pd_comparison_20261019/summary.md)。J6=50/4 在该速度下改善可重复；默认增益未自动修改，跨日旧强纹波的变化尚未定因。

测量链路诊断可显式加 `--position-probe`（仅 tracking）：独立 SocketCAN 线程以 50 Hz 只读所选轴 `0x7019` 浮点位置，一次只保留一个待响应请求。请求起止、原始回复、内核接收时间和读取时间保存为 JSONL/NPZ 的 annotation `position_probe_sample`，不参与控制或替换默认验收数据。无回复、溢出或时间戳异常会中止采集并进入既有回位流程；探针关闭在运动清理后执行。它是内部高精度通道，仍可能与状态帧共享编码器，不能称为外部角度真值。`0x701B` 为速度，`0x701A` 为 Iq 电流。

同步位置通道与固定 kp 的阻尼对照见 [2026-10-19 内部测量链路报告](realrobot_data/measurement_chain_20261019/summary.md)。浮点位置中仍有相近波动，降低 kd 后波动增大；尚未完成外部真实轴角验证。

往返使用固定起终点，默认总行程 0.55 rad、两端各 1 秒平滑变速。对所有速度使用共同的测量位置区间。开始前检查关节限位；进入 HOME 后逐轴误差需小于 0.05 rad。

采完立即离线查看实际运动：

```bash
python3 analyze_friction.py fit realrobot_data/friction_XXX.npz --output /tmp/j6_tracking --plot
```

输出 `tracking.png`、`aligned.npz`、`analysis.json`。检查实际位置是否台阶式运动、局部位置斜率是否稳定、有效掩码和分段失败原因。驱动器速度与位置斜率同时保留，不能用命令速度冒充实际速度。

默认门限是初始工程配置：局部位置拟合窗口 0.20 s、最大间隔 0.04 s、速度相对误差 15% 加 0.003 rad/s、速度 CV 15%、加速度 0.15 rad/s²、连续有效滑动至少 0.8 s、稳定样本比例 70%。这些门限需结合编码器分辨率和实机工况复核。可用 `--gates gates.json` 显式覆盖 `friction_id.analysis.Gates` 字段，修改会写入报告和参数。不得只为了通过验收而放宽门限。

tracking 模式始终不能输出可部署参数。无法稳定滑动的低速档应保留失败记录，先改善控制或提高最低辨识速度。

## 第三阶段：D 主辨识、C/E 对照与独立验证

D 需要标定后的反馈力矩。未提供标定时仍可采集和检查运动，但单位是 `protocol_nm_unverified`，不会进入可部署参数验证。

标定 JSON 字段：

```json
{
  "verified": false,
  "units": "N*m",
  "calibration_id": "填写真实标定编号",
  "evidence": "填写已知负载/外部测量、方向与比例校验记录的位置",
  "scale": [1, 1, 1, 1, 1, 1],
  "offset": [0, 0, 0, 0, 0, 0]
}
```

`tau_calibrated = scale * tau_protocol + offset`。在完成实际标定前保持 verified=false；文件只是记录证据，不会替代物理标定。当前辨识直接使用驱动器坐标，必须确认位置方向与 RS 模型一致。

```bash
python3 run_phase2_pointmatch.py --mode identify --joint 6 --load-id empty_fixed_gripper_v1 --torque-calibration calibration.json --dry-run
python3 run_phase2_pointmatch.py --mode identify --joint 6 --load-id empty_fixed_gripper_v1 --torque-calibration calibration.json --execute
python3 analyze_friction.py fit realrobot_data/friction_TRAIN.npz --output /tmp/j6_fit --gravity --plot
```

默认八档 0.05–0.20 rad/s，每档 3 次往返，速度顺序按 seed 打乱。默认不加入待辨识摩擦前馈；重力项从启动保持开始渐入。

质量检查合格后，对每次往返按位置分箱；重复编码器位置无需作为插值轴，不混入跨方向回摆。匹配其他五轴实际构型，再保存每个位置箱的 q、v_plus/minus、tau_plus/minus、通过时间和完整构型。拟合使用这些位置处的实测速度，避免先对噪声速度取绝对值再平均。

- D：`(tau_fb_plus-tau_fb_minus)/2 = Fc + Fv*(v_plus-v_minus)/2`，主方法。
- C：实际构型下 `tau_fb-g(q)`，带偏置拟合，交叉检查。`--gravity` 要求本机 URDF 哈希与采集一致。
- E：主机 PD 项配对，只作诊断。A：保留合格恒速段的 PD 均值带偏置拟合，作基线诊断；不用于参数部署。

以每次往返的配对均值作为回归观测，避免把相关位置点当独立实验。检查速度覆盖、每档独立重复、秩、缩放条件数、残差及斜率不确定性。不裁剪负系数；负斜率需查明是数据问题还是模型不适用。普通 OLS 区间仍只是诊断，不能替代不同工况的重复验证。

`fit` 只生成 candidate.json，且只有完成的 identify 运行、真实标定和通过 D 门限的数据才 eligible。另做一轮独立采集再验证：

```bash
# 采用相同载荷、标定、模型、增益及共同测量区间，改变速度顺序：
python3 run_phase2_pointmatch.py --mode identify --joint 6 --seed 101 --load-id empty_fixed_gripper_v1 --torque-calibration calibration.json --execute
python3 analyze_friction.py validate realrobot_data/friction_HOLDOUT.npz --candidate /tmp/j6_fit/candidate.json --output /tmp/j6_validation --gravity
```

验证拒绝训练集重用、原始数据完全相同、条件不一致、运动质量不合格和预测误差超限；温度差异也受限。通过后才生成 `validated_parameters.json`。每轴独立完成，补偿入口可一次加载多个轴的验证文件。条件中的 load-id 应指向可复现的夹爪/载荷/线缆配置。

## 第四阶段：补偿验证

先重力基线，随后仅加载一个轴的参数，以低比例验证；不加载的轴摩擦项为零。已移除历史 Method E 硬编码参数。

```bash
python3 11_gravity_friction_compensation.py
python3 11_gravity_friction_compensation.py --load-id empty_fixed_gripper_v1 --execute
python3 11_gravity_friction_compensation.py --parameters /tmp/j6_validation/validated_parameters.json --load-id empty_fixed_gripper_v1 --friction-scale 0.1 --execute
```

默认控制频率为 100 Hz，可在 50–200 Hz 内显式配置。做定量对照时，用同一个控制循环中的可重复轨迹，避免靠自由拖动比较：

```bash
# 基线：摩擦比例为零；包含平滑到 HOME 和单轴往返，运行完成后停止。
python3 11_gravity_friction_compensation.py --tracking-joint 6 --speed 0.10 --load-id empty_fixed_gripper_v1 --execute
# 同样轨迹，仅启用已验证 J6 参数的 10%。
python3 11_gravity_friction_compensation.py --tracking-joint 6 --speed 0.10 --load-id empty_fixed_gripper_v1 --parameters /tmp/j6_validation/validated_parameters.json --friction-scale 0.1 --execute
python3 analyze_friction.py compare realrobot_data/compensation_BASE.npz realrobot_data/compensation_TRIAL.npz --output /tmp/j6_comparison.json
```

对照要求两次运行的模型、载荷配置、控制增益、运动配置、频率及重力比例一致，输出各轴位置误差 RMS/峰值、协议反馈力矩峰值、温度、无效反馈和发送失败计数。该报告为描述性对照，不自动认证控制稳定性。轨迹模式用一致的目标位置/速度，并按目标速度计算摩擦前馈；柔顺保持模式使用滤波后的实测速度。

验证文件包含来源、条件、速度和温度范围及内容校验和。模型或 load-id 不匹配会拒绝。速度超出验证范围时不外推摩擦，在范围边缘平滑渐入/渐出；零速平滑尺度为 0.02 rad/s。黏性启用增益限制到**当前发送 kd** 的 80%；该代数限制不构成整个系统稳定性的证明。

重力、摩擦比例分开；启动前馈渐入；总前馈有逐轴幅值和变化率限制。差分速度只在新反馈上更新，并用实际接收间隔滤波。诊断不再修改速度状态。退出强制执行 HOME → 折叠零位 → 到位稳定确认 → 失能；`--return-home` 仅作为兼容参数保留，不能关闭折叠回位。回位超时不会跳到终点或直接失能。

运行日志同样保存原始反馈、命令和分项前馈。比较补偿前后相同运动的跟踪误差、峰值力矩、限幅、转向与停止附近行为，再决定是否增加比例。自由拖动手感的改善不能替代轨迹跟踪验收。

## 数据格式与旧文件

新 NPZ 为 schema 3，不依赖 pickle：

- metadata_json：运行 ID、完成/中断原因、模型哈希、源码哈希、SDK 版本、单位、载荷配置、标定和实验配置。
- feedback：每个真实收到的状态帧，不按轮询次数复制缓存。
- commands：逐轴实际发送事件。
- annotations_json：生命周期、记录状态及补偿分项等事件。

分析输出另外保存对齐后的所有状态、命令引用、有效掩码与逐点配对数据。旧 phase2 NPZ 的 all_* 是段均值，缺少原始反馈时间，分析工具会明确拒绝重新辨识。旧数据保持原样。`run_phase1_realrobot.py` 仍是历史激励实验入口，其稀疏位置和全零 tau_act 不可作为完整动力学数据，本次没有扩展到惯性参数辨识。
