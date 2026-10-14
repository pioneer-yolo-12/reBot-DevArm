#!/usr/bin/env python3
"""重力 + 摩擦力补偿演示。

使用 Pinocchio 计算 g(q) 重力前馈，叠加 Coulomb + viscous 摩擦力前馈。
松手后机械臂维持位置不掉，且有一定的阻尼感。
Ctrl+C 安全归零退出。

用法:
  python3 11_gravity_friction_compensation.py
"""

import signal
import sys
import time
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, "/home/a12/reBotArm_control_py-1")

from gravity_friction_compensation import GravityFrictionCompensation
from reBotArm_control_py.actuator import RebotArm

_running = True

def _sigint_handler(signum, frame):
    global _running
    print("\n[Ctrl+C] 准备停止...")
    _running = False

signal.signal(signal.SIGINT, _sigint_handler)

ENABLED_JOINTS: list[str] = []


def main() -> None:
    print("=" * 60)
    print("  重力 + 摩擦力补偿演示")
    print("  行为: 维持位置不掉, 松手后轻微阻尼感")
    print("  Ctrl+C 安全归零退出")
    print("=" * 60)

    rebotarm = RebotArm()

    # Phase2 辨识参数 × 0.2 (柔顺模式, 值不能太大)
    fc = [0.0478, 0.0579, 0.0891, 0.0675, 0.0673, 0.0255]  # 0.2x 缩放 (全量=[0.2389, 0.2896, 0.4457, 0.3376, 0.3363, 0.1276])
    fv = [0.0416, 0.7687, 0.511, 0.2364, 0.03, 0.0724]  # 0.2x 缩放 (全量=[0.2078, 3.8434, 2.555, 1.1819, 0.1501, 0.3618])

    # 重力缩放 (来自 rebotarm_rs.yaml)
    tau_scale = [1.0, 0.98, 0.98, 1.0, 1.0, 1.0]

    ctrl = GravityFrictionCompensation(
        rebotarm,
        kp=2.0,   # 轻柔冻结 (5.0仍太硬)
        kd=2.0,   # 增加阻尼抑制振荡
        fc=fc,
        fv=fv,
        tau_scale=tau_scale,
        transition_duration=0.5,
        enabled_joints=ENABLED_JOINTS,
        log_every=100,
    )
    ctrl.start()
    print(f"[控制循环] 运行中 @ {rebotarm.rate} Hz")
    print("-" * 60)

    try:
        while _running:
            time.sleep(0.1)
    finally:
        print("\n[停止] 关闭控制循环...")
        ctrl.end()
        print("[归零] 安全归零...")
        ctrl.safe_home()
        rebotarm.disconnect()
        print("[完成] 已安全归零并断开")


if __name__ == "__main__":
    main()
