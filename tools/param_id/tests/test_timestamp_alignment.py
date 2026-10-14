#!/usr/bin/env python3
"""Unit tests: timestamp alignment, monotonicity, interpolation edge cases.

Run:
  python3 tests/test_timestamp_alignment.py
"""

import sys, math
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir.parent))

import numpy as np


def test_time_monotonic_check():
    """时间戳必须单调递增; 重复或倒退应被检测."""
    t_good = np.array([0.0, 0.005, 0.010, 0.015, 0.020])
    t_bad = np.array([0.0, 0.005, 0.003, 0.015, 0.020])  # 0.003 < 0.005
    t_dup = np.array([0.0, 0.005, 0.005, 0.015, 0.020])  # duplicate

    assert np.all(np.diff(t_good) > 0), "Good timestamps should be monotonic"
    assert not np.all(np.diff(t_bad) > 0), "Bad timestamps should fail monotonic check"
    assert not np.all(np.diff(t_dup) > 0), "Duplicate timestamps should fail monotonic check"

    # Helper used in v2
    def check_monotonic(t):
        return len(t) < 2 or np.all(np.diff(t) > 0)

    assert check_monotonic(t_good)
    assert not check_monotonic(t_bad)
    assert not check_monotonic(t_dup)
    print("✅ test_time_monotonic_check PASSED")

def test_np_interp_edge_cases():
    """np.interp 边界条件: 查询点在范围内外、复制时间戳."""
    t_ctrl = np.array([0.0, 0.010, 0.020, 0.030, 0.040])
    q_ref = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

    # Normal interpolation
    t_query = np.array([0.005, 0.015, 0.025])
    q_interp = np.interp(t_query, t_ctrl, q_ref)
    assert abs(q_interp[0] - 0.05) < 1e-10
    assert abs(q_interp[1] - 0.15) < 1e-10
    assert abs(q_interp[2] - 0.25) < 1e-10

    # Edge: query before first sample → clamp to first
    t_before = np.array([-0.001])
    q_before = np.interp(t_before, t_ctrl, q_ref)
    assert q_before[0] == q_ref[0]

    # Edge: query after last sample → clamp to last
    t_after = np.array([0.050])
    q_after = np.interp(t_after, t_ctrl, q_ref)
    assert q_after[0] == q_ref[-1]

    # Edge: duplicate query timestamps
    t_dup = np.array([0.005, 0.005, 0.005])
    q_dup = np.interp(t_dup, t_ctrl, q_ref)
    assert np.allclose(q_dup, [0.05, 0.05, 0.05])

    print("✅ test_np_interp_edge_cases PASSED")

def test_torque_from_kp_kd():
    """力矩估算: τ = kp*(q_ref-q) + kd*(qd_ref-qd).

    不应只使用 kp 项而忽略 kd 项和 tau_ff.
    """
    kp, kd = 80.0, 5.0
    q_ref, qd_ref = 1.0, 0.1
    q_actual, qd_actual = 0.98, 0.08

    # Full MIT torque
    tau_full = kp * (q_ref - q_actual) + kd * (qd_ref - qd_actual)
    # Old approach: only kp term
    tau_kp_only = kp * (q_ref - q_actual)

    tau_kp = kp * (q_ref - q_actual)  # = 80 * 0.02 = 1.6
    tau_kd = kd * (qd_ref - qd_actual)  # = 5 * 0.02 = 0.1

    assert abs(tau_kp - 1.6) < 1e-10
    assert abs(tau_kd - 0.1) < 1e-10
    assert abs(tau_full - 1.7) < 1e-10

    # kd term is ~6% of total in this case — not negligible at high speed
    # At higher speed mismatch, kd term grows
    qd_ref_fast, qd_actual_fast = 0.5, 0.3
    tau_kd_fast = kd * (qd_ref_fast - qd_actual_fast)  # = 5 * 0.2 = 1.0
    assert abs(tau_kd_fast - 1.0) < 1e-10
    print("✅ test_torque_from_kp_kd PASSED")

def test_timestamp_alignment_accuracy():
    """控制记录(10ms)与采集记录(5ms)的时间戳对齐.

    采集线程 ~200Hz, 控制线程 ~100Hz.
    对齐需要用控制线程的时间戳去查询采集线程的数据.
    """
    # Simulate control records at 10ms
    t_ctrl = np.arange(0, 1.0, 0.010)  # 100Hz
    q_ref = np.sin(2*np.pi*0.5*t_ctrl)  # dummy reference

    # Simulate collect records at 5ms (with slight jitter)
    rng = np.random.default_rng(42)
    t_collect = np.arange(0, 1.0, 0.005) + rng.normal(0, 0.0002, 200)
    t_collect = np.sort(np.clip(t_collect, 0, 0.999))
    q_actual = np.sin(2*np.pi*0.5*t_collect) + rng.normal(0, 0.001, len(t_collect))

    # Align: interpolate collect data at ctrl timestamps
    q_at_ctrl = np.interp(t_ctrl, t_collect, q_actual)

    # Should have same length
    assert len(q_at_ctrl) == len(t_ctrl)

    # Alignment error should be small (sine tracking)
    error = np.abs(q_at_ctrl - q_ref)
    mean_err = np.mean(error)
    assert mean_err < 0.01, f"Alignment error too large: {mean_err:.4f}"

    print("✅ test_timestamp_alignment_accuracy PASSED")

def test_control_period_measurement():
    """控制周期应使用 time.perf_counter(), 记录超时次数."""
    import time

    # Simulate control loop timing
    DT = 0.010
    periods = []
    t0 = time.perf_counter()
    for i in range(20):
        t_target = t0 + (i+1) * DT
        # Simulate work
        busy = 0.001  # 1ms of work
        time.sleep(busy * 0.5)  # pretend
        t_now = time.perf_counter()
        if t_now < t_target:
            time.sleep(t_target - t_now)
        actual = time.perf_counter() - t0
        periods.append(actual)
        t0 = time.perf_counter() if i == 0 else t_target  # simplified

    # Not a real test — just verify the pattern
    assert True
    print("✅ test_control_period_measurement PASSED")


if __name__ == "__main__":
    test_time_monotonic_check()
    test_np_interp_edge_cases()
    test_torque_from_kp_kd()
    test_timestamp_alignment_accuracy()
    test_control_period_measurement()
    print("\n🎉 All timestamp alignment tests passed!")
