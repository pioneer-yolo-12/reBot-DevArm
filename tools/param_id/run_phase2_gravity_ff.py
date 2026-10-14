#!/usr/bin/env python3
"""Phase 2 v3: 轨迹连续性修复 + 力矩估算 + Pinocchio 逐点重力前馈.

与 v2 的关键区别:
  - 每次 MIT 发送时, 用 Pinocchio 计算当前关节角下的重力力矩 τ_g(q)
  - 将 τ_g(q) 作为 tau_ff 前馈 → 电机不需要用位置误差对抗重力
  - 跟踪误差大幅减小 → 摩擦力矩估算更干净
  - 重力变化不再污染 fc/fv 拟合

用法:
  python3 run_phase2_gravity_ff.py [--joint N] [--dry-run]
"""

import argparse, json, time, sys, threading
from collections import deque
from pathlib import Path
import numpy as np

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir / "mujoco_id"))
sys.path.insert(0, "/home/a12/reBotArm_control_py-1")

from trajectory import RS_LIMITS
from reBotArm_control_py.dynamics import compute_generalized_gravity
from reBotArm_control_py.kinematics.robot_model import load_robot_model

HOME = np.array([0.0, 1.2, 1.0, 0.0, 0.0, 0.0])
FOLDED = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
RATE = 100; DT = 1.0 / RATE

KP_MIT = {1: 80, 2: 150, 3: 150, 4: 25, 5: 12, 6: 25}
KD_MIT = {1: 5,  2: 10,  3: 10,  4: 1.5, 5: 0.8, 6: 1.5}
MOTOR_MODELS = ["rs-06", "rs-06", "rs-06", "rs-00", "rs-00", "rs-00"]
JOINT_NAMES = ["底座旋转", "肩部", "肘部", "腕部横滚", "腕部俯仰", "腕部旋转"]

VEL_SWEEP = [0.06, -0.06, 0.08, -0.08, 0.10, -0.10,
             0.12, -0.12, 0.15, -0.15]
T_RAMP = 1.0; D_SWEEP = 0.35; T_RET = 1.2
OUT_DIR = _this_dir / "realrobot_data"


# ── 数学工具 ──
def _quintic(t):
    t = np.clip(t, 0, 1)
    return 6*t**5 - 15*t**4 + 10*t**3

def _d_quintic(t):
    t = np.clip(t, 0, 1)
    return 30*t**4 - 60*t**3 + 30*t**2

def _quintic_vel_up(t, T, tv):
    s = np.clip(t/T, 0, 1)
    v = tv * (6*s**5 - 15*s**4 + 10*s**3)
    dq = tv * T * (s**6 - 3*s**5 + 2.5*s**4)
    return v, dq

def _quintic_vel_down(t, T, tv):
    s = np.clip(t/T, 0, 1)
    v = tv * (1 - 6*s**5 + 15*s**4 - 10*s**3)
    dq = tv * T * (s - s**6 + 3*s**5 - 2.5*s**4)
    return v, dq

def _quintic_pos_vel(t, T, start, delta):
    s = np.clip(t/T, 0, 1)
    pos = start + delta * (6*s**5 - 15*s**4 + 10*s**3)
    vel = (delta / T) * (30*s**4 - 60*s**3 + 30*s**2)
    return pos, vel


# ── Pinocchio 重力计算 ──
_pin_model = None
_pin_data = None

def get_pinocchio_gravity(q_arm):
    """计算当前 6 轴关节角下的重力力矩 [Nm]."""
    global _pin_model, _pin_data
    if _pin_model is None:
        _pin_model = load_robot_model()
        _pin_data = _pin_model.createData()
    q_full = np.zeros(_pin_model.nq)
    q_full[:6] = np.asarray(q_arm, dtype=np.float64)
    tau_g = compute_generalized_gravity(_pin_model, q_full, _pin_data)
    return np.array(tau_g[:6], dtype=np.float64)


# ── 电机组 (线程安全) ──
class MotorGroup:
    def __init__(self, motors, ctrl):
        self.motors = motors
        self.ctrl = ctrl
        self.lock = threading.Lock()
        self.fail_count = [0] * 6
        self.max_fail = 10

    def send_all_mit(self, q_ref, qd_ref, tau_ff=None):
        with self.lock:
            for j, m in enumerate(self.motors):
                try:
                    ff = float(tau_ff[j]) if tau_ff is not None else 0.0
                    m.send_mit(float(q_ref[j]), float(qd_ref[j]),
                               KP_MIT[j+1], KD_MIT[j+1], ff)
                    self.fail_count[j] = 0
                except Exception:
                    self.fail_count[j] += 1
                    if self.fail_count[j] >= self.max_fail:
                        raise RuntimeError(f"motor{j+1} send_mit连续失败!")

    def send_all_mit_safe(self, q_ref, qd_ref, tau_ff=None):
        with self.lock:
            for j, m in enumerate(self.motors):
                try:
                    ff = float(tau_ff[j]) if tau_ff is not None else 0.0
                    m.send_mit(float(q_ref[j]), float(qd_ref[j]),
                               KP_MIT[j+1], KD_MIT[j+1], ff)
                except Exception:
                    pass

    def read_position(self, j):
        return self.motors[j].robstride_get_param_f32(0x7019)

    def read_all_positions(self):
        q = np.zeros(6)
        for j in range(6):
            try: q[j] = self.read_position(j)
            except: pass
        return q

    def poll_and_get_state(self, j):
        self.ctrl.poll_feedback_once()
        return self.motors[j].get_state()

    def poll_and_get_state_safe(self, j):
        with self.lock:
            self.ctrl.poll_feedback_once()
            return self.motors[j].get_state()


# ── 采集线程 ──
class DataCollector:
    def __init__(self, mg, jidx):
        self.mg = mg; self.jidx = jidx
        self.records = []; self.running = False; self.thread = None

    def start(self):
        self.records.clear(); self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=2.0)

    def _run(self):
        while self.running:
            # 恢复 poll_feedback_once 以排空主动上报的 CAN 帧
            # 但必须通过 MotorGroup 持锁访问, 避免与 send_all_mit 并发争抢
            st = self.mg.poll_and_get_state_safe(self.jidx)
            t = time.perf_counter()
            if st is not None:
                self.records.append((t, float(st.pos), float(st.vel), float(st.torq)))
            time.sleep(0.005)


class ControlRecorder:
    def __init__(self):
        self.records = []
    def record(self, t, q_ref_j, qd_ref_j):
        self.records.append((t, float(q_ref_j), float(qd_ref_j)))
    def clear(self):
        self.records.clear()


# ── 运动原语 ──
def quintic_ramp(mg, q_start, q_target, duration):
    n = int(duration * RATE); dq = q_target - q_start
    t0 = time.perf_counter()
    for i in range(n):
        tn = (i+1)/n; a = _quintic(tn); da = _d_quintic(tn)
        q_ref = q_start + a*dq; qd_ref = (da/duration)*dq
        tau_g = get_pinocchio_gravity(q_ref)
        mg.send_all_mit(q_ref, qd_ref, tau_g)
        target_t = t0 + (i+1)*DT
        if (sl := target_t - time.perf_counter()) > 0: time.sleep(sl)

def quintic_ramp_safe(mg, q_start, q_target, duration):
    n = int(duration * RATE); dq = q_target - q_start
    t0 = time.perf_counter()
    for i in range(n):
        tn = (i+1)/n; a = _quintic(tn); da = _d_quintic(tn)
        q_ref = q_start + a*dq; qd_ref = (da/duration)*dq
        tau_g = get_pinocchio_gravity(q_ref)
        mg.send_all_mit_safe(q_ref, qd_ref, tau_g)
        target_t = t0 + (i+1)*DT
        if (sl := target_t - time.perf_counter()) > 0: time.sleep(sl)

def hold_home_mit(mg, duration):
    n = int(duration * RATE)
    tau_g = get_pinocchio_gravity(HOME)
    t0 = time.perf_counter()
    for i in range(n):
        mg.send_all_mit(HOME, np.zeros(6), tau_g)
        target_t = t0 + (i+1)*DT
        if (sl := target_t - time.perf_counter()) > 0: time.sleep(sl)

def smooth_return_to_home(mg, q_start, duration=1.5):
    """平滑 quintic 从当前位姿回到 HOME, 避免冲击."""
    n = int(duration * RATE)
    dq = HOME - q_start
    t0 = time.perf_counter()
    for i in range(n):
        tn = (i+1)/n; a = _quintic(tn); da = _d_quintic(tn)
        q_ref = q_start + a*dq
        qd_ref = (da/duration)*dq
        tau_g = get_pinocchio_gravity(q_ref)
        mg.send_all_mit(q_ref, qd_ref, tau_g)
        target_t = t0 + (i+1)*DT
        if (sl := target_t - time.perf_counter()) > 0: time.sleep(sl)


# ── 双线程扫摆 (重力前馈版) ──
def sweep_one_joint_gravity_ff(mg, jidx, motors):
    """单关节扫摆: 逐点重力前馈 + 连续轨迹."""
    jname = JOINT_NAMES[jidx]
    kp = KP_MIT[jidx+1]; kd = KD_MIT[jidx+1]

    q_home = HOME.copy()
    d_ramp = 0.0  # will be set per speed

    collector = DataCollector(mg, jidx)
    recorder = ControlRecorder()

    vel_list = []; tau_list = []
    all_v = []; all_tau = []; all_q = []
    switch_log = []

    for tv in VEL_SWEEP:
        d_ramp = tv * T_RAMP * 0.5
        d_hold = tv * (D_SWEEP / abs(tv)) if abs(tv) > 1e-6 else 0.0
        d_ret = tv * T_RET * 0.5

        T_MEAS = abs(d_hold / tv) if abs(tv) > 1e-6 else 1.0
        n_total = int((T_RAMP + T_MEAS + T_RET) * RATE)

        collector.start()
        recorder.clear()

        # Phase B→C 跳变诊断
        q_plan_b_end = None
        q_plan_c_start = None

        t0 = time.perf_counter()
        pos_errors = []; tau_est_samples = []; tau_fb_samples = []

        for step in range(n_total):
            t_elapsed = (step+1) * DT

            # ── 轨迹 ──
            if step < int(T_RAMP * RATE):
                phase = "A"
                t_i = (step+1) * DT
                v_j, dq_j = _quintic_vel_up(t_i, T_RAMP, tv)
                q_ref = q_home.copy()
                q_ref[jidx] = q_home[jidx] + dq_j
                qd_ref = np.zeros(6); qd_ref[jidx] = v_j

            elif step < int((T_RAMP + T_MEAS) * RATE):
                phase = "B"
                t_i = (step+1) * DT - T_RAMP
                q_const = q_home[jidx] + d_ramp
                q_ref = q_home.copy()
                q_ref[jidx] = q_const + tv * t_i
                qd_ref = np.zeros(6); qd_ref[jidx] = tv
                q_plan_b_end = q_ref[jidx]  # 持续更新=最后一步位置

            elif step < int((T_RAMP + T_MEAS + T_RET) * RATE):
                phase = "C"
                t_i = (step+1) * DT - T_RAMP - T_MEAS
                v_j, dq_j = _quintic_vel_down(t_i, T_RET, tv)
                q_plan_c_start = q_home[jidx] + d_ramp + d_hold + dq_j  # 持续更新=最后一步位置
                q_ref = q_home.copy()
                q_ref[jidx] = q_home[jidx] + d_ramp + d_hold + dq_j
                qd_ref = np.zeros(6); qd_ref[jidx] = v_j
            else:
                phase = "D"
                q_ref = q_home.copy()
                qd_ref = np.zeros(6)

            # ── 重力前馈 ──
            tau_g = get_pinocchio_gravity(q_ref)

            # ── 发送 MIT ──
            mg.send_all_mit(q_ref, qd_ref, tau_g)

            # ── 状态采集 ──
            st = mg.poll_and_get_state(jidx)
            if st is not None:
                act_pos = float(st.pos); act_vel = float(st.vel); act_torq = float(st.torq)
            else:
                act_pos = act_vel = act_torq = np.nan

            # ── 力矩估算 (kp+kd 项, 不含 tau_ff 因为重力已由 tau_ff 承担) ──
            tau_est = kp * (q_ref[jidx] - act_pos) + kd * (qd_ref[jidx] - act_vel)

            pos_errors.append(q_ref[jidx] - act_pos)
            tau_est_samples.append(tau_est)
            tau_fb_samples.append(act_torq)

            recorder.record(time.perf_counter(), q_ref[jidx], qd_ref[jidx])

            # ── 守时 ──
            target_t = t0 + (step+1) * DT
            if (sl := target_t - time.perf_counter()) > 0:
                time.sleep(sl)

        collector.stop()

        # ── 后处理 (恒速段数据对齐) ──
        if recorder.records:
            rt = np.array([r[0] for r in recorder.records])
            rq = np.array([r[1] for r in recorder.records])
            rd = np.array([r[2] for r in recorder.records])
        else:
            rt = rq = rd = np.array([])

        samples_v = []; samples_tau = []
        meas_start = int(T_RAMP * RATE); meas_end = int((T_RAMP + T_MEAS) * RATE)

        if collector.records:
            ct = np.array([r[0] for r in collector.records])
            cp = np.array([r[1] for r in collector.records])
            cv = np.array([r[2] for r in collector.records])
            # 时间戳单调性
            if len(ct) > 1 and not np.all(np.diff(ct) > 0):
                mask = np.concatenate([[True], np.diff(ct) > 0])
                ct = ct[mask]; cp = cp[mask]; cv = cv[mask]

            for i in range(meas_start, min(meas_end, len(tau_est_samples))):
                t_target = t0 + (i+1) * DT
                if len(ct) > 1:
                    q_at_t = np.interp(t_target, ct, cp)
                else:
                    q_at_t = act_pos
                tau_i = tau_est_samples[i]
                # 使用实际速度 (从DataCollector插值) 而非参考速度
                if len(ct) > 1:
                    v_act_at_t = float(np.interp(t_target, ct, cv))
                else:
                    v_act_at_t = float(tv)
                samples_v.append(v_act_at_t)
                samples_tau.append(float(tau_i))
                all_v.append(v_act_at_t)
                all_tau.append(float(tau_i))
                all_q.append(float(q_at_t))

        if len(samples_v) >= 20:
            vel_list.append(float(np.mean(samples_v)))
            tau_list.append(float(np.mean(samples_tau)))
            v_act_mean = float(np.mean(cv[meas_start:meas_end])) if collector.records and len(cv) > meas_end else float(tv)
            print(f"    v_cmd={tv:+.2f}  v_act={v_act_mean:+.3f}  "
                  f"τ_est={np.mean(samples_tau):+.4f}  "
                  f"τ_fb={np.mean(tau_fb_samples[meas_start:meas_end]):+.4f}  "
                  f"samples={len(samples_v)}")

        # 轨迹切换诊断
        q_jump_BC = float(q_plan_b_end - q_plan_c_start) if (q_plan_b_end is not None and q_plan_c_start is not None) else 0.0
        switch_log.append({
            "tv": float(tv), "q_jump_AB": 0.0, "q_jump_BC": q_jump_BC
        })

        # 平滑回到 HOME (避免冲击)
        q_now = mg.read_all_positions()
        smooth_return_to_home(mg, q_now, 1.5)
        hold_home_mit(mg, 0.2)

    return vel_list, tau_list, all_v, all_tau, all_q, switch_log


# ── 主入口 ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", type=int, default=0, help="关节编号 1-6 (0=全部)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 预热 Pinocchio
    print("=" * 60)
    print("  Phase 2 v3: 重力前馈 + 轨迹连续 + 力矩估算")
    print("=" * 60)
    _ = get_pinocchio_gravity(HOME)
    tau_grav_home = get_pinocchio_gravity(HOME)
    print(f"  τ_grav(HOME): {np.round(tau_grav_home, 2)}")
    print(f"  KP: {[KP_MIT[i+1] for i in range(6)]}")
    print(f"  KD: {[KD_MIT[i+1] for i in range(6)]}")

    if args.dry_run:
        print("\n[Dry-Run] 无硬件, 仅验证 Pinocchio")
        q_test = np.array([0.0, 1.55, 1.35, 0.0, 0.0, 0.0])
        tau_test = get_pinocchio_gravity(q_test)
        delta = tau_test - tau_grav_home
        print(f"  HOME→sweep 重力变化: {np.round(delta, 2)} Nm")
        print(f"  J2 重力变化: {delta[1]:.3f} Nm (这之前被错误分配给 fc)")
        return

    from motorbridge import Controller, Mode
    ctrl = Controller("can0")
    motors = [ctrl.add_robstride_motor(mid, 0xFD, mdl)
              for mid, mdl in zip(range(1, 7), MOTOR_MODELS)]
    mg = MotorGroup(motors, ctrl)

    # [1] 读取位置
    print("\n[1/5] 读取断电位置...")
    folded = mg.read_all_positions()
    print(f"  断电: {np.round(folded, 3)}")

    # [2] 初始化
    print("\n[2/5] 初始化 MIT + 主动上报...")
    for m in motors: m.clear_error(); time.sleep(0.03)
    for m in motors: m.ensure_mode(Mode.MIT, 2000)
    for m in motors: m.enable()
    tau_g_folded = get_pinocchio_gravity(folded)
    t_init = time.perf_counter()
    for i in range(int(0.3*RATE)):
        mg.send_all_mit(folded, np.zeros(6), tau_g_folded)
        target_t = t_init + (i+1)*DT
        if (sl := target_t - time.perf_counter()) > 0: time.sleep(sl)
    folded = mg.read_all_positions()
    print(f"  FOLDED(使能后): {np.round(folded, 3)}")
    t_hold = time.perf_counter()
    tau_g_folded2 = get_pinocchio_gravity(folded)
    for i in range(int(0.5*RATE)):
        mg.send_all_mit(folded, np.zeros(6), tau_g_folded2)
        target_t = t_hold + (i+1)*DT
        if (sl := target_t - time.perf_counter()) > 0: time.sleep(sl)
    for m in motors:
        try: m.robstride_set_active_report(True)
        except: pass
    print("  主动上报已开启")
    all_ok = True
    for j in range(6):
        try:
            q_test = mg.read_position(j)
            print(f"    ✓ motor{j+1} ({JOINT_NAMES[j]}) pos={q_test:+.3f}")
        except Exception as e:
            print(f"    ✗ motor{j+1} 通信失败: {e}")
            all_ok = False
    if not all_ok:
        print("  ❌ 通信校验失败")
        for m in motors:
            try: m.disable(); m.close()
            except: pass
        ctrl.close(); sys.exit(1)
    print("  全部使能 ✓")

    # [3] 展开到 HOME (带重力前馈)
    print("\n[3/5] 展开到 HOME (重力前馈)...")
    try:
        quintic_ramp(mg, folded, HOME, 3.0)
        hold_home_mit(mg, 0.5)
    except RuntimeError as e:
        print(f"\n  ❌ 展开失败: {e}")
        quintic_ramp_safe(mg, mg.read_all_positions(), folded, 3.0)
        for m in motors:
            try: m.disable(); m.close()
            except: pass
        ctrl.close(); sys.exit(1)
    q_check = mg.read_all_positions()
    for j in range(6):
        err = abs(q_check[j]-HOME[j])
        print(f"    {'✓' if err<0.2 else '✗'} joint{j+1}: {q_check[j]:+.3f} (err={err:.3f})")
    print("  到达 HOME")

    jlist = [args.joint-1] if args.joint>0 else range(6)

    # [4] 双线程扫摆 (重力前馈)
    all_data = {}; all_switch_logs = {}
    try:
        print(f"\n[4/5] 双线程扫摆 (Pinocchio 重力前馈, {len(jlist)} 关节)...")
        for jidx in jlist:
            g0 = tau_grav_home[jidx]
            print(f"\n  ── Joint{jidx+1}: {JOINT_NAMES[jidx]} (τ_grav_HOME={g0:+.3f}Nm) ──")
            vel_list, tau_list, all_v, all_tau, all_q, switch_log = \
                sweep_one_joint_gravity_ff(mg, jidx, motors)

            all_switch_logs[f"joint{jidx+1}"] = switch_log
            for sl in switch_log:
                print(f"    切换 tv={sl['tv']:+.2f}: BC跳变={sl['q_jump_BC']:.5f} (应为0)")

            if len(vel_list) >= 8:
                v_arr = np.array(vel_list); t_arr = np.array(tau_list)
                A = np.column_stack([np.ones_like(v_arr), np.sign(v_arr), v_arr])
                beta, _, _, _ = np.linalg.lstsq(A, t_arr, rcond=None)
                bias_est, fc_est, fv_est = float(beta[0]), float(beta[1]), float(beta[2])
                tau_pred = A @ beta
                rmse = float(np.sqrt(np.mean((t_arr - tau_pred)**2)))
                pos_mask = v_arr > 0; neg_mask = v_arr < 0
                rmse_pos = float(np.sqrt(np.mean((t_arr[pos_mask]-tau_pred[pos_mask])**2))) if np.any(pos_mask) else 0
                rmse_neg = float(np.sqrt(np.mean((t_arr[neg_mask]-tau_pred[neg_mask])**2))) if np.any(neg_mask) else 0
                print(f"    → 拟合: bias={bias_est:+.4f}  fc={fc_est:+.4f}  fv={fv_est:+.4f}")
                print(f"       RMSE={rmse:.4f}  RMSE(+dir)={rmse_pos:.4f}  RMSE(-dir)={rmse_neg:.4f}")
                print(f"       N={len(v_arr)}  速度范围=[{v_arr.min():+.3f}, {v_arr.max():+.3f}]")
            else:
                bias_est = fc_est = fv_est = 0.0; rmse = 0.0
                print(f"    → 数据不足, 无法拟合")

            all_data[f"joint{jidx+1}"] = {
                "name": JOINT_NAMES[jidx],
                "bias": float(bias_est), "fc": float(fc_est), "fv": float(fv_est),
                "rmse": float(rmse), "tau_grav_home": float(g0),
                "v_list": [float(x) for x in vel_list],
                "tau_list": [float(x) for x in tau_list],
                "v_samples": [float(x) for x in all_v],
                "tau_samples": [float(x) for x in all_tau],
                "q_samples": [float(x) for x in all_q],
            }
            q_now_j = mg.read_all_positions()
            smooth_return_to_home(mg, q_now_j, 1.5)
            hold_home_mit(mg, 0.2)
    except RuntimeError as e:
        print(f"\n  ❌ 扫摆异常: {e}")
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        print(f"\n[5/5] 安全折叠回零...")
        try:
            q_now = mg.read_all_positions()
            quintic_ramp_safe(mg, q_now, folded, 3.0)
            print("  已折叠")
        except:
            print("  回零异常")
            for _ in range(int(2.0*RATE)):
                for j, m in enumerate(motors):
                    try: m.send_mit(float(folded[j]), 0.0, KP_MIT[j+1], KD_MIT[j+1], 0.0)
                    except: pass
                time.sleep(DT)
        for m in motors:
            try: m.disable(); m.close()
            except: pass
        ctrl.close()
        print("  完成")
        OUT_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        np.savez(OUT_DIR / f"phase2_gravity_ff_{ts}.npz",
                 vel_sweep=VEL_SWEEP, data=all_data,
                 kp=[KP_MIT[i+1] for i in range(6)],
                 kd=[KD_MIT[i+1] for i in range(6)],
                 gravity_home=tau_grav_home.tolist(),
                 switch_logs=all_switch_logs)
        print(f"  已保存: phase2_gravity_ff_{ts}.npz")


if __name__ == "__main__":
    main()
