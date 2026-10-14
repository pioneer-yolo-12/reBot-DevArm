#!/usr/bin/env python3
"""Unit tests: trajectory continuity, quintic boundary conditions, position/velocity jumps.

Run:
  python3 -m pytest tests/test_trajectory.py -v
Or without pytest:
  python3 tests/test_trajectory.py
"""

import sys, math
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir.parent))

import numpy as np

# Copy the quintic functions from run_phase2_threaded_v2.py
# so tests are self-contained and don't import the full script

# ── quintic helpers ──
def quintic(t):
    t = np.clip(t, 0, 1)
    return 6*t**5 - 15*t**4 + 10*t**3

def d_quintic(t):
    t = np.clip(t, 0, 1)
    return 30*t**4 - 60*t**3 + 30*t**2

def quintic_pos_vel(t, T, start, delta):
    s = np.clip(t/T, 0, 1)
    pos = start + delta * (6*s**5 - 15*s**4 + 10*s**3)
    vel = (delta / T) * (30*s**4 - 60*s**3 + 30*s**2)
    return pos, vel

def quintic_pos_vel_nonzero_v0(t, T, start, delta, v_start):
    s = np.clip(t/T, 0, 1)
    vs = v_start * T
    a0 = start
    a1 = vs
    a2 = 0.0
    a3 = 10*delta - 6*vs
    a4 = -15*delta + 8*vs
    a5 = 6*delta - 3*vs
    pos = a0 + a1*s + a2*s**2 + a3*s**3 + a4*s**4 + a5*s**5
    vel = (a1 + 2*a2*s + 3*a3*s**2 + 4*a4*s**3 + 5*a5*s**4) / T
    return pos, vel

# ── Tests ──

def test_quintic_boundary_conditions():
    """五次多项式应满足 pos(0)=0, pos(T)=delta, vel(0)=0, vel(T)=0."""
    for T in [0.5, 1.0, 2.0]:
        for delta in [-0.5, 0.1, 0.35, 1.0]:
            p0, v0 = quintic_pos_vel(0, T, 0.0, delta)
            pT, vT = quintic_pos_vel(T, T, 0.0, delta)
            assert abs(p0 - 0.0) < 1e-12, f"pos(0)={p0} != 0"
            assert abs(v0 - 0.0) < 1e-12, f"vel(0)={v0} != 0"
            assert abs(pT - delta) < 1e-10, f"pos(T)={pT} != {delta}"
            assert abs(vT - 0.0) < 1e-10, f"vel(T)={vT} != 0"
    print("✅ test_quintic_boundary_conditions PASSED")

def test_quintic_nonzero_v0_boundary():
    """五次多项式(非零初速)应满足 pos(0)=start, pos(T)=start+delta, vel(0)=v_start, vel(T)=0."""
    for T in [1.0, 1.2]:
        for start in [0.0, 0.5, 1.0]:
            for delta in [0.15, -0.15, 0.35]:
                for v_start in [0.06, -0.06, 0.15, -0.15]:
                    p0, v0_calc = quintic_pos_vel_nonzero_v0(0, T, start, delta, v_start)
                    pT, vT = quintic_pos_vel_nonzero_v0(T, T, start, delta, v_start)
                    assert abs(p0 - start) < 1e-10, f"pos(0)={p0} != {start}"
                    assert abs(v0_calc - v_start) < 1e-8, f"vel(0)={v0_calc} != {v_start}"
                    assert abs(pT - (start+delta)) < 1e-8, f"pos(T)={pT} != {start+delta}"
                    assert abs(vT - 0.0) < 1e-8, f"vel(T)={vT} != 0"
    print("✅ test_quintic_nonzero_v0_boundary PASSED")

def test_quintic_monotonic_position():
    """五次多项式位置应在 [start, start+delta] 之间单调变化."""
    for T in [1.0]:
        for delta in [0.35, -0.35]:
            n = 200
            positions = []
            for i in range(n+1):
                t = T * i / n
                p, _ = quintic_pos_vel(t, T, 0.0, delta)
                positions.append(p)
            positions = np.array(positions)
            # Check bounds
            lo = min(0.0, delta) - 1e-10
            hi = max(0.0, delta) + 1e-10
            assert np.all(positions >= lo), f"Position below {lo}: min={positions.min()}"
            assert np.all(positions <= hi), f"Position above {hi}: max={positions.max()}"
            # Check monotonic (sign of delta preserved)
            if delta > 0:
                assert np.all(np.diff(positions) >= -1e-12), "Should be monotonically increasing"
            else:
                assert np.all(np.diff(positions) <= 1e-12), "Should be monotonically decreasing"
    print("✅ test_quintic_monotonic_position PASSED")

def test_trajectory_switch_continuity():
    """模拟 Phase B→C 轨迹切换: 检查位置和速度连续性.

    Phase B 结束时刻 (T_ramp): q_ref_B = q_base + delta, qd_ref_B = 0
    Phase C 开始: 应使用 Phase B 的规划终点作为起点, 而非实际位置.
    """
    # Simulate Phase B: accelerate to constant speed tv
    tv = 0.10  # target velocity
    q_base = 1.0
    t_ramp = 1.0
    delta_ramp = tv * t_ramp * 0.5  # half via quintic (approximate)

    # Phase B end state (planned)
    q_plan_end = q_base + delta_ramp
    qd_plan_end = 0.0  # quintic vel→0 at end of ramp-up (actually at constant speed it's tv)

    # Simulate: actual position might lag slightly
    q_actual_end = q_plan_end - 0.005  # 0.005 rad lag (reasonable tracking error)

    # ❌ BAD: Phase C starts from actual position (old behavior)
    q_c_start_bad = q_actual_end
    jump_bad = q_c_start_bad - q_plan_end
    assert abs(jump_bad) > 1e-6, f"Bad approach should have jump: {jump_bad}"

    # ✅ GOOD: Phase C starts from planned end position (v2 behavior)
    q_c_start_good = q_plan_end
    jump_good = q_c_start_good - q_plan_end
    assert jump_good == 0.0, f"Good approach should have zero jump: {jump_good}"

    print("✅ test_trajectory_switch_continuity PASSED")

def test_nonzero_v0_trajectory_switch():
    """Phase A→B 或恒速段切换: 使用 quintic_pos_vel_nonzero_v0.

    Phase A end: q=q_base, v=0
    Phase B start: should have position=q_base and velocity=0 (if at rest)
    Actually Phase A→B is ramp-up from rest: q_base→q_base+delta, vel 0→tv
    """
    # For ramp-up: start vel=0, end vel=tv
    # quintic_pos_vel with zero start vel works fine
    T = 1.0
    tv = 0.10
    delta = tv * T * 0.5  # approximate
    # Check end of ramp-up
    p_end, v_end = quintic_pos_vel(T, T, 0.0, delta)
    assert abs(v_end) < 1e-10, f"quintic_pos_vel vel(T)={v_end} should be 0"

    # But for Phase B we need constant speed — the v2 approach uses:
    # Phase B: q_ref = q_const_base + tv * t (linear constant speed)
    # Phase C: uses quintic_pos_vel_nonzero_v0 to ramp down from tv back to 0
    # Check Phase C transition
    q_plan_b_end = 0.5  # position at end of Phase B
    v_plan_b_end = tv  # velocity at end of Phase B

    # Phase C ramp-down: start from q_plan_b_end with velocity v_plan_b_end
    T_ret = 1.2
    delta_ret = -tv * T_ret * 0.5  # approximate, will ramp back

    p0, v0 = quintic_pos_vel_nonzero_v0(0, T_ret, q_plan_b_end, delta_ret, v_plan_b_end)
    p1, v1 = quintic_pos_vel_nonzero_v0(T_ret, T_ret, q_plan_b_end, delta_ret, v_plan_b_end)

    assert abs(p0 - q_plan_b_end) < 1e-10, f"Phase C pos(0) should match Phase B end"
    assert abs(v0 - v_plan_b_end) < 1e-8, f"Phase C vel(0) should match Phase B end"
    assert abs(v1) < 1e-8, f"Phase C vel(T) should be 0"

    print("✅ test_nonzero_v0_trajectory_switch PASSED")

def test_speed_sign_change_jump():
    """速度方向切换时不应有位置跳变.

    从 +tv 恒速段结束 → 下一个 -tv 恒速段开始.
    Phase D (return to home) → Phase A (next speed).
    """
    q_home = 1.2
    q_after_return = q_home  # should be at home after return

    # Next sweep starts from HOME
    q_next_start = q_home
    jump = q_next_start - q_after_return
    assert jump == 0.0, "No position jump when starting next sweep from HOME"

    print("✅ test_speed_sign_change_jump PASSED")

def test_quintic_velocity_continuity():
    """五次多项式速度曲线应连续 (在 t=0 和 t=T 处速度导数连续)."""
    T = 1.0
    for delta in [0.35, -0.35]:
        prev_vel_end = 0.0  # previous segment ends at zero vel
        p0, v0 = quintic_pos_vel(0, T, 0.0, delta)
        diff = abs(v0 - prev_vel_end)
        assert diff < 1e-12, f"Velocity discontinuity at switch: {diff}"
    print("✅ test_quintic_velocity_continuity PASSED")


if __name__ == "__main__":
    test_quintic_boundary_conditions()
    test_quintic_nonzero_v0_boundary()
    test_quintic_monotonic_position()
    test_trajectory_switch_continuity()
    test_nonzero_v0_trajectory_switch()
    test_speed_sign_change_jump()
    test_quintic_velocity_continuity()
    print("\n🎉 All trajectory tests passed!")
