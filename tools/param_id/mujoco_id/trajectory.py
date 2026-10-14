"""Fourier-series excitation trajectory for reBot B601-RS.

Standard form with initial condition constraints:
  q_j(t) = q0_j + Σ_{k=1}^{H} [a_jk/ω_k·sin(ω_k·t) - b_jk/ω_k·(cos(ω_k·t) - 1)]

This guarantees:
  q(0)   = q0    (no position jump!)
  q̇(0)  = 0     (enforced: Σ a_jk = 0)
  q̈(0)  = 0     (enforced: Σ k·b_jk = 0)

where ω_k = 2π·k / period.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional, Tuple

RS_LIMITS = {
    "joint1": (-2.8, 2.8),
    "joint2": (0.0, 3.14),
    "joint3": (0.0, 3.14),
    "joint4": (-1.57, 1.57),
    "joint5": (-1.57, 1.57),
    "joint6": (-3.14, 3.14),
}


@dataclass
class FourierTrajectory:
    """Multi-DOF Fourier excitation trajectory with safety constraints."""

    q0: np.ndarray
    period: float
    n_harmonics: int
    a_coeffs: np.ndarray          # [n_joints, H]
    b_coeffs: np.ndarray          # [n_joints, H]
    omega: float = field(init=False)
    joint_limits: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self):
        self.omega = 2.0 * np.pi / self.period
        self._enforce_initial_constraints()

    def _enforce_initial_constraints(self):
        """Enforce q̇(0)=0 and q̈(0)=0.

        q̇(0) = Σ a_k = 0  → adjust a_0 to cancel sum of others
        q̈(0) = Σ k·b_k = 0 → adjust b_0 to cancel weighted sum
        """
        H = self.n_harmonics
        for j in range(self.q0.shape[0]):
            # q̇(0) constraint: Σ a_k = 0
            a_sum = np.sum(self.a_coeffs[j, 1:])
            self.a_coeffs[j, 0] = -a_sum

            # q̈(0) constraint: Σ k·b_k = 0  (k = 1, 2, ..., H)
            k = np.arange(1, H + 1)
            b_weighted_sum = np.sum(self.b_coeffs[j, 1:] * k[1:])
            self.b_coeffs[j, 0] = -b_weighted_sum / k[0]  # b0 = -(Σ k·b_k)/1

    @classmethod
    def from_random(
        cls,
        q0: np.ndarray,
        period: float = 10.0,
        n_harmonics: int = 5,
        coefficient_scale: float = 0.3,
        seed: int = 42,
    ) -> "FourierTrajectory":
        rng = np.random.default_rng(seed)
        a = rng.uniform(-coefficient_scale, coefficient_scale,
                        (len(q0), n_harmonics))
        b = rng.uniform(-coefficient_scale, coefficient_scale,
                        (len(q0), n_harmonics))
        return cls(q0, period, n_harmonics, a, b)

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate (q, qd, qdd) at time t.

        q(t)  = q0 + Σ [a_k/ω_k·sin(ω_k·t) - b_k/ω_k·(cos(ω_k·t) - 1)]
        q̇(t)  = Σ [a_k·cos(ω_k·t) + b_k·sin(ω_k·t)]
        q̈(t)  = Σ [-a_k·ω_k·sin(ω_k·t) + b_k·ω_k·cos(ω_k·t)]
        """
        pos = self.q0.copy().astype(np.float64)
        vel = np.zeros_like(pos)
        acc = np.zeros_like(pos)
        H = self.n_harmonics

        for j in range(len(self.q0)):
            for h in range(H):
                order = h + 1
                wk = self.omega * order
                cos_wt = np.cos(wk * t)
                sin_wt = np.sin(wk * t)
                a_jh = self.a_coeffs[j, h]
                b_jh = self.b_coeffs[j, h]

                # Position: q0 + a/ω·sin(ωt) - b/ω·(cos(ωt) - 1)
                pos[j] += (a_jh / wk) * sin_wt - (b_jh / wk) * (cos_wt - 1.0)
                # Velocity
                vel[j] += a_jh * cos_wt + b_jh * sin_wt
                # Acceleration
                acc[j] += -a_jh * wk * sin_wt + b_jh * wk * cos_wt

        return pos, vel, acc

    def evaluate_trajectory(self, dt: float, duration: float | None = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray]:
        if duration is None:
            duration = self.period * 3
        t = np.arange(0, duration, dt)
        n = len(t)
        q = np.zeros((n, len(self.q0)))
        qd = np.zeros_like(q)
        qdd = np.zeros_like(q)
        for i in range(n):
            q[i], qd[i], qdd[i] = self.evaluate(t[i])
        return t, q, qd, qdd

    def check_safety(self, vel_limits: np.ndarray,
                     tau_limits: np.ndarray) -> bool:
        dt = 0.001
        t, q, qd, _ = self.evaluate_trajectory(dt, self.period)
        if self.joint_limits:
            for j, (lo, hi) in enumerate(self.joint_limits):
                if np.any(q[:, j] < lo - 0.01) or np.any(q[:, j] > hi + 0.01):
                    return False
        for j in range(len(self.q0)):
            if np.max(np.abs(qd[:, j])) > vel_limits[j]:
                return False
        return True

    def get_position_range(self, dt: float = 0.01):
        """Return (q_min, q_max) for safety preview."""
        t, q, _, _ = self.evaluate_trajectory(dt, self.period)
        return np.min(q, axis=0), np.max(q, axis=0)
