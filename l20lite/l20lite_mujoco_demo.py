#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent


# ============================================================
# L20 Lite 10-D control interface
#
# 0  Thumb Proximal
# 1  Thumb Side Bend
# 2  Index Proximal
# 3  Middle Proximal
# 4  Ring Proximal
# 5  Little Proximal
# 6  Index Side Bend
# 7  Ring Side Bend
# 8  Little Side Bend
# 9  Thumb Roll
# ============================================================

SDK_TO_URDF = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
    "index_mcp_roll",
    "ring_mcp_roll",
    "pinky_mcp_roll",
    "thumb_cmc_roll",
)


# ============================================================
# Presets
# ============================================================

OPEN = np.array(
    [
        0,      # thumb flex
        0,      # thumb side bend
        0,      # index flex
        0,      # middle flex
        0,      # ring flex
        0,      # pinky flex
        0,      # index side
        0,      # ring side
        0,      # pinky side
        0,      # thumb roll
    ],
    dtype=float,
)


FIST = np.array(
    [
        95,     # thumb flex
        55,     # thumb side bend
        100,    # index
        100,    # middle
        100,    # ring
        100,    # pinky
        0,      # index side
        0,      # ring side
        0,      # pinky side
        60,     # thumb roll
    ],
    dtype=float,
)


HALF = np.array(
    [
        55,
        35,
        55,
        55,
        55,
        55,
        0,
        0,
        0,
        35,
    ],
    dtype=float,
)


# ============================================================
# Parse URDF mimic relations
# ============================================================

def parse_mimic_map(
    urdf_path: Path,
):

    root = (
        ET.parse(
            urdf_path
        ).getroot()
    )

    result = {}

    for joint in root.findall(
        "joint"
    ):

        mimic = joint.find(
            "mimic"
        )

        if mimic is None:
            continue

        result[
            joint.attrib["name"]
        ] = (
            mimic.attrib["joint"],
            float(
                mimic.attrib.get(
                    "multiplier",
                    "1",
                )
            ),
            float(
                mimic.attrib.get(
                    "offset",
                    "0",
                )
            ),
        )

    return result


# ============================================================
# SDK-like angle interface
# ============================================================

class _AngleAPI:

    def __init__(
        self,
        hand,
    ):
        self._hand = hand

    def set_angles(
        self,
        values,
    ):
        self._hand.set_angles(
            values
        )

    def get_angles(
        self,
    ):
        return (
            self._hand
            .get_angles()
        )


# ============================================================
# Simulated L20 Lite
# ============================================================

class SimL20Lite:

    """
    Simulation-side API.

    Control:
        hand.angle.set_angles(
            [10 values]
        )

    Range:
        each value: 0 ~ 100

    Current simulation convention:
        0   -> URDF lower limit
        100 -> URDF upper limit

    Hardware polarity / zero offsets will
    be handled inside the real-hardware
    backend later.
    """

    def __init__(
        self,
        model,
        data,
        side,
        mimic_map,
    ):

        if side not in {
            "left",
            "right",
        }:
            raise ValueError(
                "side must be "
                "'left' or 'right'"
            )

        self.model = model
        self.data = data
        self.side = side

        self.prefix = (
            f"{side}_"
        )

        self.mimic_map = {
            k: v
            for k, v
            in mimic_map.items()
            if k.startswith(
                self.prefix
            )
        }

        self._last_cmd = (
            OPEN.copy()
        )

        self.angle = (
            _AngleAPI(self)
        )

        self._active_full_names = [
            self.prefix + name
            for name
            in SDK_TO_URDF
        ]

        # Check all active joints.
        for name in (
            self._active_full_names
        ):
            self._joint_id(
                name
            )

    # --------------------------------------------------------

    def _joint_id(
        self,
        name,
    ):

        jid = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
        )

        if jid < 0:
            raise KeyError(
                "Joint not found "
                "in MuJoCo model: "
                f"{name}"
            )

        return jid

    # --------------------------------------------------------

    def _joint_qpos_adr(
        self,
        name,
    ):

        jid = self._joint_id(
            name
        )

        return int(
            self.model
            .jnt_qposadr[
                jid
            ]
        )

    # --------------------------------------------------------

    def _joint_range(
        self,
        name,
    ):

        jid = self._joint_id(
            name
        )

        lo, hi = (
            self.model
            .jnt_range[
                jid
            ]
        )

        return (
            float(lo),
            float(hi),
        )

    # --------------------------------------------------------

    def _write_joint(
        self,
        name,
        q,
    ):

        lo, hi = (
            self._joint_range(
                name
            )
        )

        q = float(
            np.clip(
                q,
                lo,
                hi,
            )
        )

        adr = (
            self._joint_qpos_adr(
                name
            )
        )

        self.data.qpos[
            adr
        ] = q

    # --------------------------------------------------------

    def _read_joint(
        self,
        name,
    ):

        adr = (
            self._joint_qpos_adr(
                name
            )
        )

        return float(
            self.data.qpos[
                adr
            ]
        )

    # --------------------------------------------------------

    def set_angles(
        self,
        values,
    ):

        cmd = np.asarray(
            values,
            dtype=float,
        ).reshape(-1)

        if cmd.shape != (10,):

            raise ValueError(
                "L20 Lite command "
                "must contain exactly "
                "10 values. "
                f"Got {cmd.shape}"
            )

        cmd = np.clip(
            cmd,
            0.0,
            100.0,
        )

        # --------------------------------
        # 10 independent joints
        # --------------------------------

        for pct, full_name in zip(
            cmd,
            self._active_full_names,
        ):

            lo, hi = (
                self._joint_range(
                    full_name
                )
            )

            q = (
                lo
                + (pct / 100.0)
                * (hi - lo)
            )

            self._write_joint(
                full_name,
                q,
            )

        # --------------------------------
        # 10 mechanical mimic joints
        # --------------------------------

        for (
            child,
            (
                parent,
                multiplier,
                offset,
            ),
        ) in self.mimic_map.items():

            q_parent = (
                self._read_joint(
                    parent
                )
            )

            q_child = (
                multiplier
                * q_parent
                + offset
            )

            self._write_joint(
                child,
                q_child,
            )

        self._last_cmd = (
            cmd.copy()
        )

    # --------------------------------------------------------

    def get_angles(
        self,
    ):

        return (
            self._last_cmd
            .tolist()
        )


# ============================================================
# Animation functions
# ============================================================

def smoothstep(x):

    x = float(
        np.clip(
            x,
            0.0,
            1.0,
        )
    )

    return (
        x
        * x
        * (
            3.0
            - 2.0 * x
        )
    )


def run_static(
    viewer,
    model,
    data,
    left,
    right,
    pose,
):

    left.angle.set_angles(
        pose
    )

    right.angle.set_angles(
        pose
    )

    mujoco.mj_forward(
        model,
        data,
    )

    while viewer.is_running():

        viewer.sync()

        time.sleep(
            0.01
        )


def run_cycle(
    viewer,
    model,
    data,
    left,
    right,
    duration,
):

    sequence = [

        (
            OPEN,
            OPEN,
            "both open",
        ),

        (
            FIST,
            FIST,
            "both fist",
        ),

        (
            OPEN,
            FIST,
            "left open / right fist",
        ),

        (
            FIST,
            OPEN,
            "left fist / right open",
        ),

        (
            HALF,
            HALF,
            "both half-close",
        ),

        (
            OPEN,
            OPEN,
            "both open",
        ),
    ]

    current_left = (
        OPEN.copy()
    )

    current_right = (
        OPEN.copy()
    )

    idx = 0

    start = (
        time.monotonic()
    )

    announced = -1

    while viewer.is_running():

        (
            target_left,
            target_right,
            label,
        ) = sequence[idx]

        alpha = (
            (
                time.monotonic()
                - start
            )
            / duration
        )

        s = smoothstep(
            alpha
        )

        cmd_left = (
            current_left
            + s
            * (
                target_left
                - current_left
            )
        )

        cmd_right = (
            current_right
            + s
            * (
                target_right
                - current_right
            )
        )

        left.angle.set_angles(
            cmd_left
        )

        right.angle.set_angles(
            cmd_right
        )

        mujoco.mj_forward(
            model,
            data,
        )

        viewer.sync()

        if announced != idx:

            print(
                f"[POSE] {label}"
            )

            announced = idx

        if alpha >= 1.0:

            current_left = (
                target_left.copy()
            )

            current_right = (
                target_right.copy()
            )

            idx = (
                idx + 1
            ) % len(sequence)

            start = (
                time.monotonic()
            )

        time.sleep(
            0.01
        )


# ============================================================
# main
# ============================================================

def main():

    parser = (
        argparse
        .ArgumentParser()
    )

    parser.add_argument(
        "--urdf",
        type=Path,
        default=(
            ROOT
            / "generated"
            / "dual_l20lite.urdf"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "cycle",
            "open",
            "fist",
            "half",
        ],
        default="cycle",
    )

    parser.add_argument(
        "--transition",
        type=float,
        default=1.5,
        help=(
            "seconds per transition "
            "in cycle mode"
        ),
    )

    args = (
        parser.parse_args()
    )

    # --------------------------------

    if not args.urdf.exists():

        raise FileNotFoundError(
            f"{args.urdf} "
            "does not exist. "
            "Run "
            "prepare_l20lite_urdf.py "
            "first."
        )

    # --------------------------------
    # MuJoCo model
    # --------------------------------

    model = (
        mujoco.MjModel
        .from_xml_path(
            str(args.urdf)
        )
    )

    data = (
        mujoco.MjData(
            model
        )
    )

    mimic_map = (
        parse_mimic_map(
            args.urdf
        )
    )

    print(
        f"[MODEL] "
        f"nq={model.nq}, "
        f"nv={model.nv}, "
        f"nu={model.nu}, "
        f"njnt={model.njnt}"
    )

    print(
        "\n[SDK ORDER]"
    )

    for i, name in enumerate(
        SDK_TO_URDF
    ):

        print(
            f"  {i}: {name}"
        )

    # --------------------------------
    # Left / Right
    # --------------------------------

    left = SimL20Lite(
        model,
        data,
        "left",
        mimic_map,
    )

    right = SimL20Lite(
        model,
        data,
        "right",
        mimic_map,
    )

    left.angle.set_angles(
        OPEN
    )

    right.angle.set_angles(
        OPEN
    )

    mujoco.mj_forward(
        model,
        data,
    )

    # --------------------------------
    # Viewer
    # --------------------------------

    with (
        mujoco.viewer
        .launch_passive(
            model,
            data,
        )
        as viewer
    ):

        viewer.cam.lookat[:] = (
            np.array(
                [
                    0.0,
                    0.0,
                    0.11,
                ]
            )
        )

        viewer.cam.distance = (
            0.48
        )

        viewer.cam.azimuth = (
            90
        )

        viewer.cam.elevation = (
            -18
        )

        viewer.sync()

        if args.mode == "open":

            run_static(
                viewer,
                model,
                data,
                left,
                right,
                OPEN,
            )

        elif args.mode == "fist":

            run_static(
                viewer,
                model,
                data,
                left,
                right,
                FIST,
            )

        elif args.mode == "half":

            run_static(
                viewer,
                model,
                data,
                left,
                right,
                HALF,
            )

        else:

            run_cycle(
                viewer,
                model,
                data,
                left,
                right,
                max(
                    args.transition,
                    0.1,
                ),
            )


if __name__ == "__main__":
    main()
