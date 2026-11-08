from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


# ============================================================
# L20 Lite SDK-compatible 10-D order
# ============================================================

SDK_JOINT_ORDER = (
    "thumb_cmc_pitch",     # 0 thumb flex
    "thumb_cmc_yaw",       # 1 thumb side bend
    "index_mcp_pitch",     # 2 index flex
    "middle_mcp_pitch",    # 3 middle flex
    "ring_mcp_pitch",      # 4 ring flex
    "pinky_mcp_pitch",     # 5 pinky flex
    "index_mcp_roll",      # 6 index side bend
    "ring_mcp_roll",       # 7 ring side bend
    "pinky_mcp_roll",      # 8 pinky side bend
    "thumb_cmc_roll",      # 9 thumb roll
)


# ============================================================
# SDK-style angle sub-interface
# ============================================================

class _AngleAPI:

    def __init__(self, hand):
        self._hand = hand

    def set_angles(self, values):
        """
        values:
            length = 10
            range  = 0 ~ 100
        """
        self._hand.set_angles(values)

    def get_angles(self):
        """
        Return actual simulated joint positions
        normalized back to 0 ~ 100.
        """
        return self._hand.get_angles()


# ============================================================
# Simulated L20 Lite
# ============================================================

class SimL20Lite:

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        side: str,
    ):

        if side not in ("left", "right"):
            raise ValueError(
                "side must be 'left' or 'right'"
            )

        self.model = model
        self.data = data
        self.side = side

        self.angle = _AngleAPI(self)

        self.actuator_ids = []
        self.joint_ids = []
        self.qpos_adrs = []
        self.ctrl_ranges = []

        for i, joint_short_name in enumerate(
            SDK_JOINT_ORDER
        ):

            actuator_name = (
                f"{side}_a{i}_"
                f"{joint_short_name}"
            )

            joint_name = (
                f"{side}_"
                f"{joint_short_name}"
            )

            # -------------------------
            # actuator
            # -------------------------

            aid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )

            if aid < 0:
                raise KeyError(
                    f"Actuator not found: "
                    f"{actuator_name}"
                )

            # -------------------------
            # joint
            # -------------------------

            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if jid < 0:
                raise KeyError(
                    f"Joint not found: "
                    f"{joint_name}"
                )

            qadr = int(
                model.jnt_qposadr[jid]
            )

            lo, hi = (
                model.actuator_ctrlrange[aid]
            )

            self.actuator_ids.append(aid)
            self.joint_ids.append(jid)
            self.qpos_adrs.append(qadr)
            self.ctrl_ranges.append(
                (float(lo), float(hi))
            )

        self._last_command = np.zeros(
            10,
            dtype=np.float64,
        )

    # ========================================================

    def set_angles(self, values):

        cmd = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        if cmd.shape != (10,):
            raise ValueError(
                "L20 Lite command must "
                "contain exactly 10 values. "
                f"Got {cmd.shape}"
            )

        cmd = np.clip(
            cmd,
            0.0,
            100.0,
        )

        for i in range(10):

            lo, hi = (
                self.ctrl_ranges[i]
            )

            target_rad = (
                lo
                + cmd[i] / 100.0
                * (hi - lo)
            )

            aid = (
                self.actuator_ids[i]
            )

            self.data.ctrl[aid] = (
                target_rad
            )

        self._last_command[:] = cmd

    # ========================================================

    def get_angles(self):

        """
        Actual joint positions, normalized to 0~100.

        This is different from simply returning the last command:
        it tells us where the simulated hand actually is.
        """

        values = np.zeros(
            10,
            dtype=np.float64,
        )

        for i in range(10):

            q = float(
                self.data.qpos[
                    self.qpos_adrs[i]
                ]
            )

            lo, hi = (
                self.ctrl_ranges[i]
            )

            if abs(hi - lo) < 1e-12:
                pct = 0.0

            else:
                pct = (
                    100.0
                    * (q - lo)
                    / (hi - lo)
                )

            values[i] = np.clip(
                pct,
                0.0,
                100.0,
            )

        return values.tolist()

    # ========================================================

    def get_command(self):

        return (
            self._last_command
            .copy()
            .tolist()
        )

    # ========================================================

    def joint_position(
        self,
        joint_short_name: str,
    ) -> float:

        full_name = (
            f"{self.side}_"
            f"{joint_short_name}"
        )

        jid = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            full_name,
        )

        if jid < 0:
            raise KeyError(
                f"Joint not found: "
                f"{full_name}"
            )

        qadr = int(
            self.model.jnt_qposadr[jid]
        )

        return float(
            self.data.qpos[qadr]
        )


# ============================================================
# Whole MuJoCo dual-hand system
# ============================================================

class L20LiteMuJoCo:

    def __init__(
        self,
        xml_path: str | Path,
    ):

        self.xml_path = Path(
            xml_path
        ).resolve()

        self.model = (
            mujoco.MjModel
            .from_xml_path(
                str(self.xml_path)
            )
        )

        self.data = mujoco.MjData(
            self.model
        )

        self.left = SimL20Lite(
            self.model,
            self.data,
            "left",
        )

        self.right = SimL20Lite(
            self.model,
            self.data,
            "right",
        )

        self.reset()

    # ========================================================

    def reset(self):

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        self.left.angle.set_angles(
            [0] * 10
        )

        self.right.angle.set_angles(
            [0] * 10
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    # ========================================================

    def step(self, n=1):

        for _ in range(n):

            mujoco.mj_step(
                self.model,
                self.data,
            )

    # ========================================================

    @property
    def timestep(self):

        return float(
            self.model.opt.timestep
        )
