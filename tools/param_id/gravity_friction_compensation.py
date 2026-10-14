"""GravityFrictionCompensation — MIT 重力 + 摩擦力前馈控制器。

继承 GravityCompensation 的全部行为（柔顺重力补偿、增益过渡、安全归零），
额外加入 Coulomb + viscous 摩擦力前馈。

控制律（MIT 模式）：
    arm 组: 位置闭环 + g(q)×tau_scale + fc·sign(qd) + fv·qd
    gripper 组: MIT 保持当前位置

速度来源：连续两次 q 的差分（不依赖 get_velocities，避免 RS 固件读数问题）。

使用示例::

    import sys; sys.path.insert(0, '/home/a12/reBot-DevArm/tools/param_id')
    sys.path.insert(0, '/home/a12/reBotArm_control_py-1')
    from gravity_friction_compensation import GravityFrictionCompensation
    from reBotArm_control_py.actuator import RebotArm

    rebotarm = RebotArm()
    ctrl = GravityFrictionCompensation(
        rebotarm,
        fc=[0.2389, 0.2896, 0.4457, 0.3376, 0.3363, 0.1276],
        fv=[0.2078, 3.8434, 2.5550, 1.1819, 0.1501, 0.3618],  # Method A, phase2_pointmatch 2026-10-14
    )
    ctrl.start()
    ctrl.end()
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.dynamics import compute_generalized_gravity
from reBotArm_control_py.kinematics.robot_model import load_robot_model


class GravityFrictionCompensation:
    """MIT 位置闭环 + 重力前馈 + 摩擦力前馈。"""

    def __init__(
        self,
        rebotarm: RebotArm,
        *,
        kp: float = 2.0,
        kd: float = 1.0,
        fc: float | Sequence[float] = 0.0,
        fv: float | Sequence[float] = 0.0,
        enabled_joints: Sequence[str] | None = None,
        log_every: int = 0,
        tau_scale: float | Sequence[float] = 1.0,
        transition_duration: float = 0.5,
        joint_direction: float | Sequence[float] = 1.0,
        hold_kp: float | Sequence[float] | None = None,
        hold_kd: float | Sequence[float] | None = None,
    ) -> None:
        self.rebotarm = rebotarm
        self._kp = float(kp)
        self._kd = float(kd)
        self._enabled_joints = list(enabled_joints) if enabled_joints else []
        self._log_every = int(log_every)
        self._counter = 0
        self._running = False
        self._last_tau_total = np.zeros(0, dtype=np.float64)

        self._tau_scale_raw = tau_scale
        self._joint_direction_raw = joint_direction
        self._transition_duration = float(transition_duration)
        self._hold_kp_raw = hold_kp
        self._hold_kd_raw = hold_kd
        self._fc_raw = fc
        self._fv_raw = fv

        self._tau_scale: np.ndarray | None = None
        self._joint_direction: np.ndarray | None = None
        self._fc: np.ndarray | None = None
        self._fv: np.ndarray | None = None
        self._compliant_kp: np.ndarray | None = None
        self._compliant_kd: np.ndarray | None = None
        self._hold_kp: np.ndarray | None = None
        self._hold_kd: np.ndarray | None = None
        self._transition_q_hold: np.ndarray | None = None
        self._transition_started_at: float | None = None
        self._q_frozen: np.ndarray | None = None
        self._q_prev: np.ndarray | None = None
        self._t_prev: float | None = None
        self._gc_model = None
        self._gc_data = None

    @property
    def last_tau(self) -> np.ndarray:
        return self._last_tau_total.copy()

    def start(self) -> None:
        if self._running:
            raise RuntimeError("已在运行，请先调用 end()")

        robot = self.rebotarm
        robot.connect()

        n = robot.arm.num_joints
        self._resolve_params(n)

        q_hold = robot.arm.get_positions().copy()

        robot.arm.mode_mit()
        if robot.has_gripper:
            robot.gripper.mode_mit()
        robot.disable_all()
        time.sleep(0.1)

        if self._enabled_joints:
            for name in self._enabled_joints:
                if name in robot._motor_map:
                    robot._motor_map[name].enable()
            print(f"[安全模式] 仅使能: {self._enabled_joints}")
        else:
            robot.enable_all()
            print("[使能] 全部电机已使能")

        self._q_prev = q_hold.copy()
        self._t_prev = time.perf_counter()
        self._qd_filt = np.zeros(robot.arm.num_joints)

        self._transition_q_hold = q_hold.copy()
        self._q_frozen = None
        self._transition_started_at = time.perf_counter()
        self._running = True
        robot.start_control_loop(self._loop_cb)
        print(f"[控制循环] 启动 @ {robot.rate} Hz")
        # 启动诊断: 确认重力补偿非零
        time.sleep(0.3)
        q_test = robot.arm.get_positions()
        tau_test = self._total_torque(q_test)
        print(f"[诊断] 当前位置: {np.round(q_test, 3)}")
        print(f"[诊断] 重力+摩擦力矩: {np.round(tau_test, 3)} N·m")
        print(f"[诊断] 重力缩放: {np.round(self._tau_scale, 3)}")
        print(f"[诊断] 摩擦 fc: {np.round(self._fc, 3)}  fv: {np.round(self._fv, 3)}")

    def end(self) -> None:
        if not self._running:
            return
        self.rebotarm.stop_control_loop()
        self._running = False

    def __enter__(self) -> GravityFrictionCompensation:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.end()

    def _total_torque(self, q: np.ndarray) -> np.ndarray:
        n = len(q)

        q_model = np.zeros(self._gc_model.nq)
        q_model[:n] = q
        tau_model = compute_generalized_gravity(
            self._gc_model, q_model, self._gc_data
        )[:n]
        tau_g = tau_model * self._joint_direction * self._tau_scale

        now = time.perf_counter()
        if self._q_prev is not None and self._t_prev is not None:
            dt_actual = max(now - self._t_prev, 1e-4)
            qd_est = (q - self._q_prev) / dt_actual
        else:
            dt_actual = 0.01  # 默认 100Hz
            qd_est = np.zeros(n)

        self._q_prev = q.copy()
        self._t_prev = now

        # 低通滤波速度估计 (消除 500Hz 差分噪声, 截止 ~3Hz)
        _alpha = min(dt_actual / 0.05, 1.0)
        self._qd_filt = self._qd_filt + _alpha * (qd_est - self._qd_filt)

        _deadzone = 0.02  # 滤波后噪声已消除, 恢复正常阈值
        tau_fric = (self._fc * np.tanh(self._qd_filt / _deadzone) + self._fv * self._qd_filt) * self._tau_scale
        return tau_g + tau_fric

    def _loop_cb(self, robot: RebotArm, dt: float) -> None:
        del dt
        q = robot.arm.get_positions()
        n = robot.arm.num_joints

        tau = self._total_torque(q)
        self._last_tau_total = np.asarray(tau, dtype=np.float64)

        transition_started_at = self._transition_started_at
        elapsed = (
            self._transition_duration
            if transition_started_at is None
            else max(0.0, time.perf_counter() - transition_started_at)
        )
        blend = self._transition_blend(elapsed)
        kp = self._hold_kp + blend * (self._compliant_kp - self._hold_kp)
        kd = self._hold_kd + blend * (self._compliant_kd - self._hold_kd)
        q_hold = self._transition_q_hold
        q_target = (
            q.copy()
            if q_hold is None
            else q_hold + blend * (q - q_hold)
        )
        if blend >= 1.0:
            if self._q_frozen is None:
                self._q_frozen = q.copy()
                print(f"[冻结] 初始锁定 @ {np.round(self._q_frozen, 3)}")
            # 连续追踪: 冻结点持续缓慢跟随实际位置
            # 速度噪声通过低通滤波消除, kd=2 抑制振荡
            _track_alpha = 0.015  # τ ~ 0.13s (500Hz)
            # 微小位移不追踪 (避免编码器噪声导致漂移)
            _dq = np.linalg.norm(q - self._q_frozen)
            if _dq > 0.005:
                self._q_frozen = self._q_frozen + _track_alpha * (q - self._q_frozen)
            q_target = self._q_frozen
            self._transition_q_hold = None

        robot.arm.send_mit(
            pos=q_target,
            vel=np.zeros(n),
            kp=kp,
            kd=kd,
            tau=self._last_tau_total,
        )
        if robot.has_gripper:
            robot.gripper.send_mit(robot.gripper.get_positions())

        self._counter += 1
        if self._log_every > 0 and self._counter % self._log_every == 0:
            print(
                f"[{self._counter:4d}] "
                f"tau = " + "  ".join(f"{t:+.3f}" for t in self._last_tau_total) + "  N·m"
            )

    def safe_home(
        self,
        max_vel: float = 0.5,
        send_freq: float = 50.0,
        settle: float = 0.5,
        timeout: float = 15.0,
    ) -> None:
        if self._hold_kp is None:
            raise RuntimeError("safe_home() 需先调用 start()")
        robot = self.rebotarm
        q_hold = robot.arm.get_positions().copy()
        n = robot.arm.num_joints
        home = np.zeros(n)
        kp = self._hold_kp
        kd = self._hold_kd

        tau_hold = self._total_torque(q_hold)
        robot.arm.send_mit(q_hold, vel=np.zeros(n), kp=kp, kd=kd, tau=tau_hold)
        if robot.has_gripper:
            robot.gripper.send_mit(robot.gripper.get_positions())
        time.sleep(0.3)

        q_err = np.abs(home - q_hold)
        max_err = float(np.max(q_err))
        if max_err < 0.01:
            return

        t_ramp = max_err / max_vel
        t_total = t_ramp * 2.0
        dt_send = 1.0 / send_freq
        num_steps = max(2, int(t_total / dt_send))
        interval = t_total / num_steps

        deadline = time.monotonic() + timeout
        print(f"[safe_home] 归零: {num_steps}步 @ {send_freq:.0f}Hz, {t_total:.1f}s")
        for i in range(num_steps):
            if time.monotonic() > deadline:
                print("[safe_home] 超时")
                break
            s = (i + 1) / num_steps
            q_traj = q_hold + (home - q_hold) * (
                10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5
            )
            tau_traj = self._total_torque(q_traj)
            robot.arm.send_mit(q_traj, vel=np.zeros(n), kp=kp, kd=kd, tau=tau_traj)
            if robot.has_gripper:
                robot.gripper.send_mit(robot.gripper.get_positions())
            time.sleep(interval)

        robot.arm.send_mit(home, vel=np.zeros(n), kp=kp, kd=kd, tau=np.zeros(n))
        time.sleep(settle)

    def _resolve_params(self, n: int) -> None:
        self._tau_scale = np.ones(n) * self._tau_scale_raw
        self._joint_direction = np.ones(n) * self._joint_direction_raw
        self._fc = np.ones(n) * self._fc_raw
        self._fv = np.ones(n) * self._fv_raw
        self._compliant_kp = np.full(n, self._kp)
        self._compliant_kd = np.full(n, self._kd)

        if self._hold_kp_raw is None:
            self._hold_kp = self.rebotarm.arm._mit_kp
        else:
            self._hold_kp = np.ones(n) * self._hold_kp_raw
        if self._hold_kd_raw is None:
            self._hold_kd = self.rebotarm.arm._mit_kd
        else:
            self._hold_kd = np.ones(n) * self._hold_kd_raw

        self._gc_model = load_robot_model()
        self._gc_data = self._gc_model.createData()

    def _transition_blend(self, elapsed: float) -> float:
        ratio = float(np.clip(
            elapsed / max(self._transition_duration, 1e-6), 0.0, 1.0))
        return ratio * ratio * (3.0 - 2.0 * ratio)
