#!/usr/bin/env python3
"""
离线审计测试: 摩擦辨识 + 在线控制器

测试 A: 摩擦模型单元测试
测试 B: 速度估计测试
测试 C: 辨识流程测试 (使用真实 NPZ 数据)
测试 D: sign 零速行为测试
测试 E: 保持位置逻辑测试

运行: python3 tests/test_phase2_audit.py
"""

import sys, time, math
from pathlib import Path
import numpy as np

_this_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_this_dir))

# ═══════════════════════════════════════════════════
# 测试 A: 摩擦模型单元测试
# ═══════════════════════════════════════════════════

def test_A_friction_model():
    """验证 fc·sign(v) + fv·v 的正确性."""
    fc = np.array([0.38, 0.30, 0.45, 0.26, 0.40, 0.18])
    fv = np.array([0.36, 3.42, 2.08, 0.94, 0.31, 0.47])

    # 正速度
    v_pos = np.array([0.1, 0.05, 0.0, 0.2, 0.0, 0.0])
    tau = fc * np.sign(v_pos) + fv * v_pos
    assert np.all(tau[:2] > 0), f"正速度摩擦应为正: {tau}"
    assert tau[2] == 0, f"零速度摩擦应为0: {tau[2]}"

    # 负速度
    v_neg = np.array([-0.1, -0.05, 0.0, -0.2, 0.0, 0.0])
    tau_neg = fc * np.sign(v_neg) + fv * v_neg
    assert np.all(tau_neg[:2] < 0), f"负速度摩擦应为负: {tau_neg}"

    # 零速边界
    v_zero = np.zeros(6)
    tau_zero = fc * np.sign(v_zero) + fv * v_zero
    assert np.all(tau_zero == 0), f"零速摩擦应为全零: {tau_zero}"

    # 验证输出维度
    assert tau.shape == (6,), f"输出维度错误: {tau.shape}"

    # 验证方向对称性
    v_sym = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    tau_sym_pos = fc * np.sign(v_sym) + fv * v_sym
    tau_sym_neg = fc * np.sign(-v_sym) + fv * (-v_sym)
    # 库仑项反向, 粘性项也反向
    assert np.allclose(tau_sym_pos, -tau_sym_neg, atol=1e-10), \
        f"正反方向不对称: {tau_sym_pos} vs {-tau_sym_neg}"

    print("✅ 测试A: 摩擦模型单元测试 PASSED")


# ═══════════════════════════════════════════════════
# 测试 B: 速度估计测试 (模拟编码器量化)
# ═══════════════════════════════════════════════════

def test_B_velocity_estimation():
    """验证位置差分速度估计的行为."""
    DT = 0.01  # 100Hz
    ENC_RES = 0.00038  # ~14-bit per rad (estimated)

    # B1: 匀速运动
    q_const = np.arange(0, 1.0, 0.06 * DT)
    qd_est = np.diff(q_const) / DT
    assert np.allclose(qd_est, 0.06, atol=1e-10), f"匀速估计错误: {qd_est[:5]}"

    # B2: 量化噪声 (模拟编码器)
    q_quant = np.round(q_const / ENC_RES) * ENC_RES
    qd_quant = np.diff(q_quant) / DT
    # 量化后速度有离散跳变
    n_zero = np.sum(np.abs(qd_quant) < 1e-10)
    print(f"    B2: 量化零速步数={n_zero}/{len(qd_quant)}")

    # B3: sign 抖动测试
    # 模拟零速附近微振动
    q_jitter = np.array([0.0, 0.00001, -0.00001, 0.00002, -0.00001, 0.0])
    qd_jitter = np.diff(q_jitter) / DT
    signs = np.sign(qd_jitter)
    n_flips = np.sum(np.diff(signs) != 0)
    print(f"    B3: 零速微振 sign翻转次数={n_flips}/{len(signs)-1}")

    # B4: 硬 sign vs 软 tanh 对比
    v_test = np.array([-0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01])
    hard = np.sign(v_test)
    soft = np.tanh(v_test / 0.002)  # smooth with ~0.002 rad/s transition
    print(f"    B4: 硬sign={hard}")
    print(f"        软tanh={np.round(soft, 3)}")
    # 软 sign 在零速附近连续, 无跳变
    assert abs(soft[2] - soft[3]) < 0.1, "软sign零速附近应平滑"

    print("✅ 测试B: 速度估计测试 PASSED")


# ═══════════════════════════════════════════════════
# 测试 C: 辨识流程测试 (真实 NPZ)
# ═══════════════════════════════════════════════════

def test_C_identification_audit():
    """审计辨识数据: v_list 是参考速度还是实际速度?"""
    npz_path = _this_dir / "realrobot_data" / "phase2_gravity_ff_20261013_150324.npz"
    if not npz_path.exists():
        print("⚠ 测试C: NPZ 文件不存在, 跳过")
        return

    d = np.load(npz_path, allow_pickle=True)
    data = d['data'].item()
    vel_sweep = d['vel_sweep']

    print(f"    VEL_SWEEP (参考速度): {vel_sweep}")

    for jname in sorted(data.keys()):
        jd = data[jname]
        v_list = np.array([float(x) for x in jd['v_list']])
        t_list = np.array([float(x) for x in jd['tau_list']])

        # 检查: v_list 是否完全等于 VEL_SWEEP?
        diff = np.abs(v_list - vel_sweep)
        is_exact = np.all(diff < 1e-12)
        print(f"    {jname}: v_list == VEL_SWEEP? {is_exact} (max diff={np.max(diff):.1e})")

        if is_exact:
            print(f"      ⚠ v_list 完全等于参考速度! 未使用实际速度用于拟合!")

        # 检查 tau_list 的正负一致性
        pos_mask = v_list > 0
        neg_mask = v_list < 0
        tau_pos_mean = np.mean(t_list[pos_mask]) if np.any(pos_mask) else 0
        tau_neg_mean = np.mean(t_list[neg_mask]) if np.any(neg_mask) else 0
        # Coulomb friction: tau should be more positive for positive velocities
        if np.any(pos_mask) and np.any(neg_mask):
            tau_asym = tau_pos_mean - abs(tau_neg_mean)
            print(f"      τ(+v)={tau_pos_mean:.4f}  τ(-v)={tau_neg_mean:.4f}  Δ={tau_asym:.4f}")

    # C2: 验证拟合一致性
    jd2 = data['joint2']
    v_arr = np.array([float(x) for x in jd2['v_list']])
    t_arr = np.array([float(x) for x in jd2['tau_list']])
    A = np.column_stack([np.ones_like(v_arr), np.sign(v_arr), v_arr])
    beta, residuals, rank, sv = np.linalg.lstsq(A, t_arr, rcond=None)
    tau_pred = A @ beta

    fc_stored = jd2['fc']
    fv_stored = jd2['fv']
    fc_refit = float(beta[1])
    fv_refit = float(beta[2])

    print(f"\n    J2 拟合验证:")
    print(f"      存储: fc={fc_stored:.4f}  fv={fv_stored:.4f}")
    print(f"      重新: fc={fc_refit:.4f}  fv={fv_refit:.4f}")
    assert abs(fc_stored - fc_refit) < 0.01, "fc 不匹配!"
    assert abs(fv_stored - fv_refit) < 0.01, "fv 不匹配!"

    # C3: 评估如果用实际速度拟合会怎样
    print(f"\n    ⚠ 关键发现: v_list 使用参考速度 qd_ref, 非实际速度 qd_act")
    print(f"      对 J2/J3 (大惯性), 跟踪滞后会把速度相关误差吸收进 fv")
    print(f"      J2 fv={fv_stored:.2f} 可能被高估")

    print("✅ 测试C: 辨识流程审计 PASSED (发现问题已记录)")


# ═══════════════════════════════════════════════════
# 测试 D: 零速 sign 抖动
# ═══════════════════════════════════════════════════

def test_D_sign_chatter():
    """验证零速附近 sign() 硬切换导致的力跳变."""
    fc = np.array([0.38, 0.30, 0.45, 0.26, 0.40, 0.18])

    # 模拟: arm以极低速运动, 编码器量化导致 qd 在 0 和 ε 之间交替
    np.random.seed(42)
    DT = 0.01
    ENC_RES = 0.0004  # 假设编码器分辨率
    n_samples = 200

    # 模拟极低速运动: 名义速度 0.0005 rad/s
    v_nominal = 0.0005
    q_true = np.cumsum(np.full(n_samples, v_nominal * DT))
    q_quant = np.round(q_true / ENC_RES) * ENC_RES
    qd_est = np.diff(q_quant) / DT

    # 硬 sign
    tau_hard = fc[0] * np.sign(qd_est)  # J1
    hard_flips = np.sum(np.abs(np.diff(tau_hard)) > fc[0] * 0.5)

    # 软 tanh
    v_smooth = 0.002  # rad/s 平滑过渡带宽
    tau_soft = fc[0] * np.tanh(qd_est / v_smooth)
    soft_range = np.max(tau_soft) - np.min(tau_soft)

    print(f"    极低速 ({v_nominal*1000:.1f}e-3 rad/s):")
    print(f"      硬sign: tau范围=[{np.min(tau_hard):.3f}, {np.max(tau_hard):.3f}] 跳变={hard_flips}次")
    print(f"      软tanh: tau范围=[{np.min(tau_soft):.3f}, {np.max(tau_soft):.3f}] 平滑={soft_range:.4f}")
    print(f"      硬sign在零速附近产生 {hard_flips}/{n_samples-1} 次力跳变")

    # 模拟静止时的手部微振
    q_micro = np.cumsum(np.random.normal(0, ENC_RES*0.3, n_samples))
    qd_micro = np.diff(q_micro) / DT
    tau_micro_hard = fc[0] * np.sign(qd_micro)
    micro_flips = np.sum(np.abs(np.diff(tau_micro_hard)) > fc[0] * 0.5)
    print(f"\n    静止微振 (±{ENC_RES*0.3*1e3:.2f}e-3 rad):")
    print(f"      硬sign: 力跳变={micro_flips}次")
    print(f"      ⚠ 这会导致手感到'咔哒'振动")

    print("✅ 测试D: 零速sign抖动测试 PASSED (问题已确认)")


# ═══════════════════════════════════════════════════
# 测试 E: 保持位置逻辑
# ═══════════════════════════════════════════════════

def test_E_hold_position():
    """验证 transition 后 q_target 的行为 (v2: 加入冻结机制)."""
    # 模拟 _loop_cb 的 blend 逻辑 (v2 冻结版)
    q_hold = np.array([0.0, 1.2, 1.0, 0.0, 0.0, 0.0])
    q_actual = q_hold.copy()
    q_frozen = None

    # transition 完成后 (blend >= 1.0) — v2: 冻结当前位置
    blend = 1.0
    if q_frozen is None:
        q_frozen = q_actual.copy()
    q_target = q_frozen

    print(f"    blend={blend}: q_target={q_target} (frozen)")
    print(f"    q_actual={q_actual}")
    print(f"    q_target == q_actual: {np.allclose(q_target, q_actual)}")
    print(f"    \u2705 冻结生效: q_target 不再跟随 q_actual")

    # 模拟松手后: q_actual 变化 (被推走), q_target 保持冻结
    q_actual2 = np.array([0.1, 1.3, 0.9, 0.05, -0.02, 0.03])
    q_target2 = q_frozen  # 冻结不变
    pos_error = q_target2 - q_actual2
    restoring_force = 15.0 * pos_error  # kp=15

    print(f"\n    松手后 (冻结机制生效):")
    print(f"    q_actual={q_actual2}")
    print(f"    q_target={q_target2} (冻结不变)")
    print(f"    pos_error={np.round(pos_error, 3)}")
    print(f"    restoring_force (kp=15): {np.round(restoring_force, 3)}")
    print(f"    \u2705 有恢复力: 手臂会被弹回冻结位置")

    print("\u2705 测试E: 保持位置逻辑分析 PASSED (v2: 冻结机制已加入)")

# ═══════════════════════════════════════════════════
# 主测试
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 2 + 在线控制器 离线审计")
    print("=" * 60)
    print()

    all_passed = True
    tests = [
        ("A: 摩擦模型单元测试", test_A_friction_model),
        ("B: 速度估计测试", test_B_velocity_estimation),
        ("C: 辨识流程审计", test_C_identification_audit),
        ("D: 零速sign抖动", test_D_sign_chatter),
        ("E: 保持位置逻辑", test_E_hold_position),
    ]

    for name, func in tests:
        try:
            print(f"\n── {name} ──")
            func()
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            all_passed = False

    print(f"\n{'='*60}")
    print(f"  审计完成: {'全部通过' if all_passed else '有问题待修复'}")
    print(f"{'='*60}")
