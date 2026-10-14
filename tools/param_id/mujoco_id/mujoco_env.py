"""MuJoCo simulation wrapper for reBot B601-RS parameter identification.

Compatible with MuJoCo >= 3.10.
"""

from pathlib import Path
import mujoco
import numpy as np

_MODEL_DIR = Path(
    "/home/a12/ReBot_Arm_DigitalTwin_RS/rebotarm_ros2_RS/src/"
    "rebotarm_mujoco_rs/models"
)
_ARM_XML = _MODEL_DIR / "rs_arm.xml"

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
N_ARM = 6


class ReBotMjEnv:
    """Minimal MuJoCo environment for the reBot RS arm."""

    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path) if model_path else _ARM_XML
        if not path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {path}")

        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        # Fix: disable parent-child self-collision between base_link and link1
        # (contype=1 overrides MuJoCo's default parent-child exclusion)
        for g in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[g]
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name in ('base_link', 'link1') and self.model.geom_contype[g] > 0:
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0

        # Joint dof addresses
        self._dof_addr = np.zeros(N_ARM, dtype=int)
        for i in range(N_ARM):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                    ARM_JOINTS[i])
            if jid < 0:
                raise ValueError(f"Joint {ARM_JOINTS[i]} not found")
            self._dof_addr[i] = self.model.jnt_dofadr[jid]

        # Actuator IDs (by joint name: joint1_motor ... joint6_motor)
        self._act_ids = np.zeros(N_ARM, dtype=int)
        for i in range(N_ARM):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                    f"joint{i+1}_motor")
            if aid < 0:
                raise ValueError(f"Actuator joint{i+1}_motor not found")
            self._act_ids[i] = aid

    def reset(self, q0: np.ndarray | None = None):
        mujoco.mj_resetData(self.model, self.data)
        if q0 is not None:
            self.data.qpos[self._dof_addr] = np.asarray(q0, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self._dof_addr].copy()

    @property
    def qd(self) -> np.ndarray:
        return self.data.qvel[self._dof_addr].copy()

    @property
    def qdd(self) -> np.ndarray:
        return self.data.qacc[self._dof_addr].copy()

    @property
    def tau(self) -> np.ndarray:
        return self.data.ctrl[self._act_ids].copy()

    @property
    def qfrc_bias(self) -> np.ndarray:
        return self.data.qfrc_bias[self._dof_addr].copy()

    @property
    def mass_matrix(self) -> np.ndarray:
        """Return full 9x9 mass matrix via mj_fullM (MuJoCo >= 3.0 API)."""
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, M)
        return M

    def step(self, tau_cmd: np.ndarray | None = None):
        if tau_cmd is not None:
            self.data.ctrl[self._act_ids] = np.asarray(tau_cmd, dtype=np.float64)
        mujoco.mj_step(self.model, self.data)

    def get_state(self) -> dict:
        return {"q": self.q, "qd": self.qd, "qdd": self.qdd, "tau": self.tau}

    @property
    def arm_body_ids(self) -> list:
        """Return body IDs for the 7 main arm bodies (base_link through link6)."""
        bodies = []
        for name in ["base_link"] + [f"link{i}" for i in range(1, 7)]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                bodies.append(bid)
        return bodies

    @staticmethod
    def find_model() -> Path:
        if _ARM_XML.exists():
            return _ARM_XML
        raise FileNotFoundError(f"RS MuJoCo model not found at {_ARM_XML}")
