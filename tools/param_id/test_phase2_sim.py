#!/usr/bin/env python3
"""
Phase2 摩擦辨识 — MuJoCo 仿真验证 (v8: 逐点位置匹配 vs 段平均)
  新增方法 D: 文档严格逐点匹配 [τ_out(q,+v) - τ_ret(q,-v)] / 2
  保留方法 B: 段平均对称
  保留方法 C: tau_fb - tau_g
"""

import sys, time, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/a12/reBotArm_control_py-1")

import mujoco
from mujoco import viewer as mujoco_viewer
from mujoco_id.mujoco_env import ReBotMjEnv, ARM_JOINTS, N_ARM
from reBotArm_control_py.kinematics.robot_model import load_robot_model
from pinocchio import computeGeneralizedGravity

_pin_model = load_robot_model()
_pin_data = _pin_model.createData()

def _gravity(q6):
    q = np.zeros(_pin_model.nq); q[:6] = q6
    return computeGeneralizedGravity(_pin_model, _pin_data, q)[:6]

FC_TRUE = np.array([0.30, 0.40, 0.35, 0.25, 0.35, 0.15])
FV_TRUE = np.array([1.00, 2.50, 2.50, 0.80, 0.25, 0.50])

HOME = np.array([0.0, 1.2, 1.0, 0.0, 0.0, 0.0])
RATE = 500; DT = 1.0/RATE

KP_MIT = {1: 80, 2: 150, 3: 150, 4: 25, 5: 25, 6: 25}
KD_MIT = {1: 5, 2: 10, 3: 10, 4: 1.5, 5: 1.5, 6: 1.5}

VEL_ABS = [0.06, 0.08, 0.10, 0.12, 0.15]
T_RAMP = 1.0; D_SWEEP = 0.35; T_RET = 1.2


def _quintic_vel_up(t, T, tv):
    s = np.clip(t/T, 0, 1)
    v = tv*(6*s**5 - 15*s**4 + 10*s**3)
    dq = tv*T*(s**6 - 3*s**5 + 2.5*s**4)
    return v, dq

def _quintic_vel_down(t, T, tv):
    s = np.clip(t/T, 0, 1)
    v = tv*(1 - 6*s**5 + 15*s**4 - 10*s**3)
    dq = tv*T*(s - s**6 + 3*s**5 - 2.5*s**4)
    return v, dq

def _quintic(t):
    t = np.clip(t, 0, 1)
    return 6*t**5 - 15*t**4 + 10*t**3


class SimMotorGroup:
    def __init__(self, env, viewer):
        self.env = env; self.viewer = viewer
        self._step_count = 0

    def send_all_mit(self, q_ref, qd_ref, tau_ff):
        q = self.env.q.copy(); qd = self.env.qd.copy()
        tau = np.zeros(6)
        for j in range(6):
            tau[j] = KP_MIT[j+1]*(q_ref[j]-q[j]) + KD_MIT[j+1]*(qd_ref[j]-qd[j]) + tau_ff[j]
        self.env.data.ctrl[self.env._act_ids] = tau
        mujoco.mj_step(self.env.model, self.env.data)
        self._step_count += 1
        if self.viewer and self._step_count % 5 == 0:
            self.viewer.sync()

    def get_state(self, jidx):
        q = self.env.q.copy(); qd = self.env.qd.copy()
        return float(q[jidx]), float(qd[jidx]), float(self.env.data.actuator_force[self.env._act_ids[jidx]])

    def read_all_positions(self):
        return self.env.q.copy()


def quintic_ramp_home(mg, q_start, q_target, duration):
    n = int(duration * RATE)
    t0 = time.perf_counter()
    for i in range(n):
        tn = (i+1)/n
        a = _quintic(tn)
        qr = q_start + a*(q_target - q_start)
        qdr = (q_target - q_start) * (30*tn**4 - 60*tn**3 + 30*tn**2) / duration
        mg.send_all_mit(qr, qdr, _gravity(qr))
        target_t = t0 + (i+1)*DT
        now = time.perf_counter()
        if now < target_t:
            time.sleep(target_t - now)


def collect_const_vel_raw(mg, jidx, tv, q_start_pos, kp, kd, q_home):
    """采集恒速段逐点原始数据: (positions, velocities, tau_est, tau_fb, tau_grav)."""
    d_ramp_total = abs(tv) * T_RAMP * 0.5
    d_hold_total = abs(tv) * (D_SWEEP / abs(tv)) if abs(tv) > 1e-6 else 0.0
    T_MEAS = abs(d_hold_total / tv) if abs(tv) > 1e-6 else 1.0
    n_total = int((T_RAMP + T_MEAS + T_RET) * RATE)
    i_meas_start = int(T_RAMP * RATE)
    i_meas_end = int((T_RAMP + T_MEAS) * RATE)

    pos_list, vel_list, te_list, tf_list, tg_list = [], [], [], [], []

    t0 = time.perf_counter()
    for step in range(n_total):
        t_elapsed = (step+1) * DT

        if step < int(T_RAMP * RATE):
            v_j, dq_j = _quintic_vel_up((step+1)*DT, T_RAMP, tv)
            q_ref = q_home.copy(); qd_ref = np.zeros(6)
            q_ref[jidx] = q_start_pos + dq_j; qd_ref[jidx] = v_j
        elif step < int((T_RAMP + T_MEAS) * RATE):
            t_i = (step+1)*DT - T_RAMP
            d_ramp_this = tv * T_RAMP * 0.5
            q_ref = q_home.copy(); qd_ref = np.zeros(6)
            q_ref[jidx] = q_start_pos + d_ramp_this + tv * t_i; qd_ref[jidx] = tv
        elif step < n_total:
            t_i = (step+1)*DT - T_RAMP - T_MEAS
            v_j, dq_j = _quintic_vel_down(t_i, T_RET, tv)
            q_ref = q_home.copy(); qd_ref = np.zeros(6)
            q_ref[jidx] = q_start_pos + d_ramp_this + d_hold_total + dq_j
            qd_ref[jidx] = v_j
        else:
            q_ref = q_home.copy(); qd_ref = np.zeros(6)

        tau_g = _gravity(q_ref)
        mg.send_all_mit(q_ref, qd_ref, tau_g)

        cpos, cvel, ctorq = mg.get_state(jidx)
        tau_est = kp*(q_ref[jidx] - cpos) + kd*(qd_ref[jidx] - cvel)

        if i_meas_start <= step < i_meas_end:
            pos_list.append(cpos)
            vel_list.append(cvel)
            te_list.append(tau_est)
            tf_list.append(ctorq)
            tg_list.append(tau_g[jidx])

        target_t = t0 + (step+1)*DT
        now = time.perf_counter()
        if now < target_t:
            time.sleep(target_t - now)

    return (np.array(pos_list), np.array(vel_list), np.array(te_list),
            np.array(tf_list), np.array(tg_list))


def sweep_one_joint(mg, jidx):
    kp = KP_MIT[jidx+1]; kd = KD_MIT[jidx+1]
    q_home = HOME.copy()

    # 存储往返逐点数据
    out_raw = {}  # abs_v -> (pos, vel, te, tf, tg)
    ret_raw = {}

    for abs_v in VEL_ABS:
        # --- Outbound ---
        pos, vel, te, tf, tg = collect_const_vel_raw(mg, jidx, +abs_v, q_home[jidx], kp, kd, q_home)
        out_raw[abs_v] = (pos, vel, te, tf, tg)
        v_mean = float(np.mean(vel))
        print(f"    out +{abs_v:.2f}: pos=[{pos[0]:.3f}..{pos[-1]:.3f}] v={v_mean:+.4f} "
              f"te={np.mean(te):+.4f} tf={np.mean(tf):+.4f} tg={np.mean(tg):+.3f} N={len(pos)}")

        # Hold at +D
        d_ramp = abs_v * T_RAMP * 0.5
        d_hold = abs_v * (D_SWEEP / abs_v)
        d_ret = abs_v * T_RET * 0.5
        q_at_d = q_home.copy()
        q_at_d[jidx] = q_home[jidx] + d_ramp + d_hold + d_ret
        for _ in range(int(0.3 * RATE)):
            mg.send_all_mit(q_at_d, np.zeros(6), _gravity(q_at_d))
            time.sleep(DT)

        # --- Return ---
        pos, vel, te, tf, tg = collect_const_vel_raw(mg, jidx, -abs_v, q_at_d[jidx], kp, kd, q_home)
        ret_raw[abs_v] = (pos, vel, te, tf, tg)
        v_mean = float(np.mean(vel))
        print(f"    ret -{abs_v:.2f}: pos=[{pos[0]:.3f}..{pos[-1]:.3f}] v={np.mean(vel):+.4f} "
              f"te={np.mean(te):+.4f} tf={np.mean(tf):+.4f} tg={np.mean(tg):+.3f} N={len(pos)}")

        # 回 HOME
        q_now = mg.read_all_positions()
        quintic_ramp_home(mg, q_now, q_home, 1.5)

    # ── 方法 B: 段平均对称 ──
    results = {}
    v_seg, tau_seg = [], []
    for abs_v in VEL_ABS:
        _, vel_o, _, tf_o, _ = out_raw[abs_v]
        _, vel_r, _, tf_r, _ = ret_raw[abs_v]
        v_avg = (abs(np.mean(vel_o)) + abs(np.mean(vel_r))) / 2.0
        tf_diff = (np.mean(tf_o) - np.mean(tf_r)) / 2.0
        v_seg.append(v_avg); tau_seg.append(tf_diff)

    if len(v_seg) >= 2:
        B = np.column_stack([np.ones_like(v_seg), v_seg])
        beta, _, _, _ = np.linalg.lstsq(B, tau_seg, rcond=None)
        results['B_seg_avg'] = dict(fc=float(beta[0]), fv=float(beta[1]),
                                     rmse=float(np.sqrt(np.mean((tau_seg - B@beta)**2))),
                                     desc='段平均: [τ_out-τ_ret]/2')

    # ── 方法 D: 文档逐点位置匹配 ──
    v_pt, tau_pt = [], []
    for abs_v in VEL_ABS:
        pos_o, vel_o, _, tf_o, _ = out_raw[abs_v]
        pos_r, vel_r, _, tf_r, _ = ret_raw[abs_v]

        # 找出 outbound 和 return 位置重叠区间
        # outbound: pos 从小到大, return: pos 从大到小
        q_min = max(pos_o[0], pos_r[-1])   # 重叠区下界
        q_max = min(pos_o[-1], pos_r[0])   # 重叠区上界

        if q_max <= q_min:
            # 无重叠
            continue

        # 在重叠区内均匀采样位置点
        n_pts = min(200, len(pos_o)//5, len(pos_r)//5)
        q_samples = np.linspace(q_min, q_max, n_pts)

        # 在 outbound 和 return 的原始数据中插值
        # return 数据位置是递减的，需要 flip 来做 interp
        tf_o_at_q = np.interp(q_samples, pos_o, tf_o)
        tf_r_at_q = np.interp(q_samples, pos_r[::-1], tf_r[::-1])  # flip to increasing

        # 逐点力矩差
        tau_diff_per_point = (tf_o_at_q - tf_r_at_q) / 2.0
        tau_diff_mean = float(np.mean(tau_diff_per_point))

        v_o_at_q = np.interp(q_samples, pos_o, vel_o)
        v_r_at_q = np.interp(q_samples, pos_r[::-1], vel_r[::-1])
        v_avg_per_point = (np.abs(v_o_at_q) + np.abs(v_r_at_q)) / 2.0
        v_mean = float(np.mean(v_avg_per_point))

        v_pt.append(v_mean)
        tau_pt.append(tau_diff_mean)

        # 同时计算 tau_fb - tau_g 的方法
        tf_net_o = tf_o_at_q - np.interp(q_samples, pos_o, tg_list_o) if False else None

    if len(v_pt) >= 2:
        B = np.column_stack([np.ones_like(v_pt), v_pt])
        beta, _, _, _ = np.linalg.lstsq(B, tau_pt, rcond=None)
        results['D_point_match'] = dict(fc=float(beta[0]), fv=float(beta[1]),
                                         rmse=float(np.sqrt(np.mean((tau_pt - B@beta)**2))),
                                         desc='逐点匹配: [τ_out(q)-τ_ret(q)]/2 (文档)')

    # ── 方法 C: tau_fb - tau_g (段平均) ──
    all_v_c, all_tau_c = [], []
    for abs_v in VEL_ABS:
        _, vel_o, _, tf_o, tg_o = out_raw[abs_v]
        _, vel_r, _, tf_r, tg_r = ret_raw[abs_v]
        all_v_c.append(float(np.mean(vel_o)))
        all_tau_c.append(float(np.mean(tf_o - tg_o)))
        all_v_c.append(float(np.mean(vel_r)))
        all_tau_c.append(float(np.mean(tf_r - tg_r)))

    if len(all_v_c) >= 4:
        A = np.column_stack([np.ones_like(all_v_c), np.sign(all_v_c), all_v_c])
        beta, _, _, _ = np.linalg.lstsq(A, all_tau_c, rcond=None)
        results['C_fb_minus_g'] = dict(fc=float(beta[1]), fv=float(beta[2]),
                                        bias=float(beta[0]),
                                        rmse=float(np.sqrt(np.mean((all_tau_c - A@beta)**2))),
                                        desc='tau_fb - tau_g + bias')

        A_nb = np.column_stack([np.sign(all_v_c), all_v_c])
        beta_nb, _, _, _ = np.linalg.lstsq(A_nb, all_tau_c, rcond=None)
        results['C_no_bias'] = dict(fc=float(beta_nb[0]), fv=float(beta_nb[1]),
                                     bias=0.0,
                                     rmse=float(np.sqrt(np.mean((all_tau_c - A_nb@beta_nb)**2))),
                                     desc='tau_fb - tau_g (无bias)')

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    print("="*70)
    print("  Phase2 MuJoCo Validation (v8: 逐点匹配 vs 段平均)")
    print("="*70)
    print(f"  True fc: {FC_TRUE},  True fv: {FV_TRUE}")

    XML = "/home/a12/ReBot_Arm_DigitalTwin_RS/rebotarm_ros2_RS/src/rebotarm_mujoco_rs/models/rs_arm_with_scene.xml"
    env = ReBotMjEnv(XML)

    for j in range(N_ARM):
        dof = env._dof_addr[j]
        env.model.dof_frictionloss[dof] = FC_TRUE[j]
        env.model.dof_damping[dof] = FV_TRUE[j]

    viewer = None
    if not args.headless:
        viewer = mujoco_viewer.launch_passive(env.model, env.data)

    env.reset(HOME)
    mujoco.mj_forward(env.model, env.data)
    mg = SimMotorGroup(env, viewer)

    all_results = {}
    for jidx in range(6):
        q_now = mg.read_all_positions()
        if not np.allclose(q_now, HOME, atol=0.01):
            quintic_ramp_home(mg, q_now, HOME, 2.0)

        print(f"\n  -- Joint{jidx+1} --")
        results = sweep_one_joint(mg, jidx)
        all_results[jidx] = results
        for method, r in results.items():
            print(f"    {method:18s}: fc={r['fc']:.4f}  fv={r['fv']:.4f}  "
                  f"rmse={r['rmse']:.4f}  [{r['desc']}]")

    # ── 汇总对比 ──
    methods = ['B_seg_avg', 'D_point_match', 'C_fb_minus_g', 'C_no_bias']
    print("\n" + "="*70)
    print("  fc 对比")
    print("="*70)
    h = f"  {'Jt':<4} {'true':>6}"
    for m in methods: h += f" {m:>16}"
    print(h)
    print("  " + "-"*(10+17*len(methods)))
    for j in range(6):
        line = f"  J{j+1:<3} {FC_TRUE[j]:6.2f}"
        for m in methods:
            r = all_results.get(j, {}).get(m, {})
            line += f" {r.get('fc',0):16.4f}"
        print(line)

    print(f"\n  fv 对比")
    print(f"  {'Jt':<4} {'true':>6}", end="")
    for m in methods: print(f" {m:>16}", end="")
    print()
    print("  " + "-"*(10+17*len(methods)))
    for j in range(6):
        line = f"  J{j+1:<3} {FV_TRUE[j]:6.2f}"
        for m in methods:
            r = all_results.get(j, {}).get(m, {})
            line += f" {r.get('fv',0):16.4f}"
        print(line)

    # 误差
    print(f"\n  fc MAE:")
    for m in methods:
        errs = [abs(all_results.get(j,{}).get(m,{}).get('fc',0)-FC_TRUE[j]) for j in range(6)]
        print(f"    {m:18s}: {np.mean(errs):.4f}")
    print(f"  fv MAE:")
    for m in methods:
        errs = [abs(all_results.get(j,{}).get(m,{}).get('fv',0)-FV_TRUE[j]) for j in range(6)]
        print(f"    {m:18s}: {np.mean(errs):.4f}")

    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
