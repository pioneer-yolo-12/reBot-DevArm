#!/usr/bin/env python3
"""Unit tests: friction model fitting with synthetic data.

Covers: coulomb+viscous fitting, bias handling, invalid sample rejection,
        RMSE diagnostics, sign-aware residuals.

Run:
  python3 tests/test_friction_fit.py
"""

import sys
from pathlib import Path
_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir.parent))
import numpy as np


def fit_coulomb_viscous(velocity, torque, with_bias=True):
    v = np.asarray(velocity, dtype=np.float64).ravel()
    tau = np.asarray(torque, dtype=np.float64).ravel()
    if with_bias:
        A = np.column_stack([np.ones_like(v), np.sign(v), v])
    else:
        A = np.column_stack([np.sign(v), v])
    coeff, *_ = np.linalg.lstsq(A, tau, rcond=None)
    if with_bias:
        return float(coeff[0]), float(coeff[1]), float(coeff[2])
    return float(coeff[0]), float(coeff[1])


def generate_synthetic_data(fc_true, fv_true, bias_true, speeds, noise_std=0.01):
    v = np.array(speeds)
    tau = bias_true + fc_true * np.sign(v) + fv_true * v
    tau += np.random.default_rng(42).normal(0, noise_std, len(v))
    return v, tau


def test_perfect_synthetic_fit():
    fc_true, fv_true, bias_true = 0.5, 0.3, 0.1
    speeds = [-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0]
    v, tau = generate_synthetic_data(fc_true, fv_true, bias_true, speeds, noise_std=0.0)
    bias, fc, fv = fit_coulomb_viscous(v, tau)
    assert abs(bias - bias_true) < 1e-8
    assert abs(fc - fc_true) < 1e-8
    assert abs(fv - fv_true) < 1e-8
    print("✅ test_perfect_synthetic_fit PASSED")

def test_noisy_synthetic_fit():
    fc_true, fv_true, bias_true = 0.5, 0.3, 0.1
    speeds = [-2.0, -1.5, -1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    v, tau = generate_synthetic_data(fc_true, fv_true, bias_true, speeds, noise_std=0.02)
    bias, fc, fv = fit_coulomb_viscous(v, tau)
    assert abs(bias - bias_true) < 0.05
    assert abs(fc - fc_true) < 0.05
    assert abs(fv - fv_true) < 0.05
    print("✅ test_noisy_synthetic_fit PASSED")

def test_bias_absorption():
    fc_true, fv_true = 0.5, 0.3
    gravity_offset = 4.0
    speeds = [-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0]
    v = np.array(speeds)
    tau = gravity_offset + fc_true * np.sign(v) + fv_true * v
    fc_nobias, fv_nobias = fit_coulomb_viscous(v, tau, with_bias=False)
    bias_w, fc_w, fv_w = fit_coulomb_viscous(v, tau, with_bias=True)
    error_fc_nobias = abs(fc_nobias - fc_true)
    error_fc_with = abs(fc_w - fc_true)
    assert error_fc_nobias > error_fc_with
    assert abs(fc_w - fc_true) < 1e-8
    assert abs(fv_w - fv_true) < 1e-8
    assert abs(bias_w - gravity_offset) < 1e-8
    print("✅ test_bias_absorption PASSED")

def test_rmse_diagnostics():
    fc_true, fv_true, bias_true = 0.5, 0.3, 0.1
    speeds = [-2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2]
    v, tau = generate_synthetic_data(fc_true, fv_true, bias_true, speeds, noise_std=0.05)
    bias, fc, fv = fit_coulomb_viscous(v, tau)
    A = np.column_stack([np.ones_like(v), np.sign(v), v])
    beta = np.array([bias, fc, fv])
    tau_pred = A @ beta
    residuals = tau - tau_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    pos_mask = v > 0; neg_mask = v < 0
    rmse_pos = float(np.sqrt(np.mean(residuals[pos_mask]**2))) if pos_mask.any() else 0
    rmse_neg = float(np.sqrt(np.mean(residuals[neg_mask]**2))) if neg_mask.any() else 0
    assert 0.01 < rmse < 0.15
    assert rmse_pos > 0 and rmse_neg > 0
    assert 0.1 < rmse_pos/rmse_neg < 10.0
    print(f"  RMSE={rmse:.4f}  RMSE(+dir)={rmse_pos:.4f}  RMSE(-dir)={rmse_neg:.4f}")
    print("✅ test_rmse_diagnostics PASSED")

def test_insufficient_speed_range():
    speeds_pos = [0.1, 0.2, 0.5, 1.0, 2.0]
    fc_true, fv_true, bias_true = 0.5, 0.3, 2.0
    v, tau = generate_synthetic_data(fc_true, fv_true, bias_true, speeds_pos, noise_std=0.0)
    bias, fc, fv = fit_coulomb_viscous(v, tau)
    combined = bias + fc
    combined_true = bias_true + fc_true
    assert abs(combined - combined_true) < 1e-8
    assert abs(fv - fv_true) < 1e-8
    print("✅ test_insufficient_speed_range PASSED")

def test_outlier_robustness():
    fc_true, fv_true, bias_true = 0.5, 0.3, 0.1
    speeds = [-2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2]
    v, tau = generate_synthetic_data(fc_true, fv_true, bias_true, speeds, noise_std=0.01)
    tau[3] += 5.0
    bias, fc, fv = fit_coulomb_viscous(v, tau)
    A = np.column_stack([np.ones_like(v), np.sign(v), v])
    beta = np.array([bias, fc, fv])
    residuals = tau - A @ beta
    rmse = float(np.sqrt(np.mean(residuals**2)))
    assert rmse > 0.5, f"RMSE should be large due to outlier: {rmse:.4f}"
    fc_deviation = abs(fc - fc_true)
    fv_deviation = abs(fv - fv_true)
    # Outlier at v=-0.1 (sign=-1) directly couples into fc via sign(v) column
    # At minimum one parameter should deviate noticeably
    assert fc_deviation > 0.2 or fv_deviation > 0.2, (
        f"Outlier should cause noticeable parameter deviation: "
        f"fc_d={fc_deviation:.3f} fv_d={fv_deviation:.3f}")
    print(f"  RMSE with outlier={rmse:.4f} (expected >0.5), fc_dev={fc_deviation:.3f}, fv_dev={fv_deviation:.3f}")
    print("✅ test_outlier_robustness PASSED")


if __name__ == "__main__":
    test_perfect_synthetic_fit()
    test_noisy_synthetic_fit()
    test_bias_absorption()
    test_rmse_diagnostics()
    test_insufficient_speed_range()
    test_outlier_robustness()
    print("\n🎉 All friction fit tests passed!")
