#!/usr/bin/env python3
"""Phase 1 真机: GA激励轨迹 MIT 跟踪 + 数据采集

修复:
  - RS-00 电机用低 KP/KD (10/0.6, 8/0.5)
  - 位置读数用 robstride_get_param_f32(0x7019) 代替 get_state()
  - motor 6 型号 rs-00
  - 加入重力前馈
"""
import argparse, time, sys
from pathlib import Path
import numpy as np

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir / "mujoco_id"))
sys.path.insert(0, str(_this_dir))

from trajectory import FourierTrajectory, RS_LIMITS

HOME = np.array([0.0, 1.2, 1.0, 0.0, 0.0, 0.0])
PERIOD = 10.0; N_HARMONICS = 5; DURATION = 30.0
RATE = 100; DT = 1.0 / RATE

# KP/KD: RS-06 高增益, RS-00 低增益 (参考 pd_sweep_id.py)
KP_MIT = {1: 80, 2: 150, 3: 150, 4: 10, 5: 8, 6: 10}
KD_MIT = {1: 5,  2: 10,  3: 10,  4: 0.6, 5: 0.5, 6: 0.6}
# motor 6 也是 rs-00 (参考 pd_sweep_id.py MODELS)
MOTOR_MODELS = ["rs-06", "rs-06", "rs-06", "rs-00", "rs-00", "rs-00"]
MECH_POS = 0x7019
CKPT_DIR = _this_dir / "ga_checkpoints"
OUT_DIR = _this_dir / "realrobot_data"


def _quintic(t):
    t = np.clip(t, 0, 1)
    return 6*t**5 - 15*t**4 + 10*t**3


def load_ga_trajectory():
    ckpts = sorted(CKPT_DIR.glob("ga_gen*.npz"))
    if not ckpts: return None
    ckpt = np.load(ckpts[-1])
    chrom = ckpt["chromosome"]
    n = len(chrom)//2
    a = chrom[:n].reshape(6,N_HARMONICS)*0.5
    b = chrom[n:].reshape(6,N_HARMONICS)*0.5
    traj = FourierTrajectory(HOME, PERIOD, N_HARMONICS, a, b)
    print(f"  GA: {ckpts[-1].name} cond={ckpt.get('cond','?')}")
    _, q_pre, qd_pre, _ = traj.evaluate_trajectory(0.01, DURATION)
    print(f"  峰值速度: {np.round(np.max(np.abs(qd_pre),axis=0),2)} rad/s")
    return traj


def read_q6(motors):
    """通过 mechPos 读取 6 轴位置."""
    q = np.zeros(6)
    for j, m in enumerate(motors):
        try:
            q[j] = m.robstride_get_param_f32(MECH_POS)
        except: pass
    return q


def build_full_trajectory(folded, speed, traj):
    n_traj = int(DURATION / speed / DT)
    q_traj = np.zeros((n_traj, 6)); qd_traj = np.zeros((n_traj, 6))
    for i in range(n_traj):
        t = min(i * DT * speed, DURATION - 1e-10)
        q_ref, qd_ref, _ = traj.evaluate(t)
        q_traj[i] = q_ref; qd_traj[i] = qd_ref

    n_ramp = int(3.0 / DT)
    ri = folded + np.array([_quintic((i+1)/n_ramp) for i in range(n_ramp)]).reshape(-1,1)*(HOME-folded)
    ro = HOME + np.array([_quintic((i+1)/n_ramp) for i in range(n_ramp)]).reshape(-1,1)*(folded-HOME)
    q_full = np.vstack([ri, q_traj, ro])
    qd_full = np.zeros_like(q_full); qd_full[1:-1] = (q_full[2:]-q_full[:-2])/(2*DT)
    return q_full, qd_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    print("="*60)
    print("  Phase 1 真机: GA激励轨迹 MIT 跟踪")
    print(f"  KP: {[KP_MIT[i+1] for i in range(6)]}")
    print(f"  KD: {[KD_MIT[i+1] for i in range(6)]}")
    print(f"  Models: {MOTOR_MODELS}")
    print("="*60)

    from motorbridge import Controller, Mode, CallError
    ctrl = Controller("can0")
    motors = [ctrl.add_robstride_motor(mid, 0xFD, mdl) for mid, mdl in zip(range(1, 7), MOTOR_MODELS)]

    # 1. 读当前位置
    print("\n[1/5] 读取当前关节位置...")
    if args.dry_run:
        folded = np.zeros(6)
    else:
        folded = read_q6(motors)
        if np.all(folded == 0):
            print("  ❌ 读取失败")
            return
    print(f"  FOLDED: {np.round(folded, 3)}")

    # 2. 生成轨迹
    print(f"\n[2/5] 生成轨迹...")
    traj = load_ga_trajectory()
    if traj is None: return
    q_full, qd_full = build_full_trajectory(folded, args.speed, traj)
    if q_full is None: return

    n_total = len(q_full)
    print(f"  步数: {n_total}  时长: {n_total/RATE:.0f}s")

    limits = list(RS_LIMITS.values())
    ok = True
    for j in range(6):
        lo, hi = limits[j]
        qmn, qmx = q_full[:,j].min(), q_full[:,j].max()
        in_range = qmn >= lo-0.02 and qmx <= hi+0.02
        print(f"    joint{j+1}: [{qmn:.3f},{qmx:.3f}] lim[{lo},{hi}] {'OK' if in_range else '!!'}")
        if not in_range: ok = False
    if not ok: return

    if args.dry_run:
        print("\n  Dry run 完成"); return

    if not args.yes:
        ans = input(f"\n  确认驱动机器人? (yes/no): ")
        if ans.lower() != "yes":
            print("  取消"); return

    # 3. 初始化 MIT
    print(f"\n[3/5] 初始化 MIT 模式...")
    for m in motors:
        m.clear_error(); time.sleep(0.03)
    for m in motors:
        m.ensure_mode(Mode.MIT, 2000)
    for m in motors:
        m.enable()
    print("  全部使能")
    time.sleep(0.5)

    log_q = np.zeros((n_total, 6)); log_tau = np.zeros((n_total, 6))

    try:
        print(f"\n[4/5] 执行轨迹...")
        t0 = time.time()
        for i in range(n_total):
            q_ref, qd_ref = q_full[i], qd_full[i]
            for j, m in enumerate(motors):
                try:
                    m.send_mit(
                        float(q_ref[j]), float(qd_ref[j]),
                        KP_MIT[j+1], KD_MIT[j+1], 0.0)
                except CallError: pass

            # 每 10 步读一次位置
            if i % 10 == 0 or i == n_total-1:
                q6 = read_q6(motors)
                for j in range(6):
                    log_q[i, j] = q6[j]
                
                if i % 500 == 0 or i == n_total-1:
                    print(f"    {i}/{n_total} [{time.time()-t0:.0f}s] q={np.round(q6,3)}")

            target_t = t0 + (i+1)*DT
            if (sl := target_t - time.time()) > 0: time.sleep(sl)

    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        print(f"\n[5/5] 电机关闭...")
        for m in motors:
            try: m.disable(); m.close()
            except: pass
        ctrl.close()
        print("  完成")

        OUT_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        np.savez(OUT_DIR / f"phase1_real_{ts}.npz",
                 q_ref=q_full, qd_ref=qd_full,
                 q_act=log_q, tau_act=log_tau,
                 kp=[KP_MIT[i+1] for i in range(6)],
                 kd=[KD_MIT[i+1] for i in range(6)])
        print(f"  已保存: phase1_real_{ts}.npz")


if __name__ == "__main__":
    main()
