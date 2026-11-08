#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np

from linkerbot import L20lite


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from l20lite_sim import L20LiteMuJoCo
from l20lite_joint_mapping import (
    real_sdk_to_mujoco,
)



MODEL_PATH = (
    PROJECT_ROOT
    / "generated"
    / "dual_l20lite_dynamic.xml"
)

REAL_SIDE = "right"
CAN_INTERFACE = "can0"

STEP_TIME = 0.05
MOVE_TIME = 1.5


# ============================================================
# Helpers
# ============================================================

def smoothstep(x):
    x = np.clip(
        x,
        0.0,
        1.0,
    )

    return (
        x * x
        * (3.0 - 2.0 * x)
    )


def send_pose(
    hand,
    sim,
    q_real,
):
    """
    q_real is always expressed in REAL LinkerBot SDK space.

    The exact same semantic pose goes to:

        REAL hand directly

        MuJoCo after
        real_sdk_to_mujoco(...)
    """

    q_real = np.clip(
        np.asarray(
            q_real,
            dtype=np.float64,
        ),
        0.0,
        100.0,
    )

    q_sim = real_sdk_to_mujoco(
        q_real
    )

    # --------------------------
    # REAL
    # --------------------------

    hand.angle.set_angles(
        q_real.tolist()
    )

    # --------------------------
    # MuJoCo RIGHT
    # --------------------------

    sim.right.angle.set_angles(
        q_sim
    )

    # Left simulated hand unused.
    sim.left.angle.set_angles(
        [0.0] * 10
    )


def move_pose(
    hand,
    sim,
    viewer,
    q_start,
    q_target,
    duration=MOVE_TIME,
):

    q_start = np.asarray(
        q_start,
        dtype=np.float64,
    )

    q_target = np.asarray(
        q_target,
        dtype=np.float64,
    )

    steps = max(
        1,
        int(
            duration
            / STEP_TIME
        ),
    )

    sim_steps = max(
        1,
        int(
            STEP_TIME
            / sim.timestep
        ),
    )

    for k in range(
        steps + 1
    ):

        if not viewer.is_running():
            return

        alpha = (
            k / steps
        )

        s = smoothstep(
            alpha
        )

        q = (
            q_start
            + s
            * (
                q_target
                - q_start
            )
        )

        send_pose(
            hand,
            sim,
            q,
        )

        for _ in range(
            sim_steps
        ):
            sim.step()

        viewer.sync()

        time.sleep(
            STEP_TIME
        )


def print_pose(
    name,
    q_real,
):

    q_sim = real_sdk_to_mujoco(
        q_real
    )

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        "REAL:",
        np.round(
            q_real,
            1,
        ).tolist(),
    )

    print(
        "SIM :",
        np.round(
            q_sim,
            1,
        ).tolist(),
    )


# ============================================================
# Build comparison poses
# ============================================================

def build_poses(q0):

    q0 = np.asarray(
        q0,
        dtype=np.float64,
    )

    poses = []

    # --------------------------------------------------------
    # 1. Baseline / open
    # --------------------------------------------------------

    OPEN = q0.copy()

    poses.append(
        (
            "1. BASE / OPEN",
            OPEN,
        )
    )

    # --------------------------------------------------------
    # 2. Four fingers half close
    #
    # Real SDK flex:
    # decreasing = more flexion
    # --------------------------------------------------------

    HALF = q0.copy()

    HALF[2:6] = np.clip(
        HALF[2:6] - 12.0,
        0.0,
        100.0,
    )

    poses.append(
        (
            "2. FOUR-FINGER HALF CLOSE",
            HALF,
        )
    )

    # --------------------------------------------------------
    # 3. Moderate close
    #
    # Intentionally NOT extreme fist.
    # --------------------------------------------------------

    CLOSE = q0.copy()

    CLOSE[2:6] = np.clip(
        CLOSE[2:6] - 20.0,
        0.0,
        100.0,
    )

    poses.append(
        (
            "3. FOUR-FINGER MODERATE CLOSE",
            CLOSE,
        )
    )

    # --------------------------------------------------------
    # 4. Index only
    # --------------------------------------------------------

    INDEX = q0.copy()

    INDEX[2] = np.clip(
        INDEX[2] - 15.0,
        0.0,
        100.0,
    )

    poses.append(
        (
            "4. INDEX FLEX",
            INDEX,
        )
    )

    # --------------------------------------------------------
    # 5. Finger spread
    #
    # index/ring/pinky ABD increase.
    # --------------------------------------------------------

    SPREAD = q0.copy()

    SPREAD[6] = np.clip(
        SPREAD[6] + 12.0,
        0.0,
        100.0,
    )

    SPREAD[7] = np.clip(
        SPREAD[7] + 12.0,
        0.0,
        100.0,
    )

    SPREAD[8] = np.clip(
        SPREAD[8] + 12.0,
        0.0,
        100.0,
    )

    poses.append(
        (
            "5. FINGER SPREAD",
            SPREAD,
        )
    )

    # --------------------------------------------------------
    # 6. Thumb opposition-like pose
    #
    # Moderate changes only.
    #
    # thumb_flex decrease -> toward palm
    # thumb_abd increase  -> toward index
    # --------------------------------------------------------

    THUMB_OPPOSE = q0.copy()

    THUMB_OPPOSE[0] = np.clip(
        THUMB_OPPOSE[0] - 12.0,
        0.0,
        100.0,
    )

    THUMB_OPPOSE[1] = np.clip(
        THUMB_OPPOSE[1] + 10.0,
        0.0,
        100.0,
    )

    poses.append(
        (
            "6. THUMB OPPOSE",
            THUMB_OPPOSE,
        )
    )

    # --------------------------------------------------------
    # 7. Thumb yaw
    # --------------------------------------------------------

    THUMB_YAW = q0.copy()

    THUMB_YAW[9] = np.clip(
        THUMB_YAW[9] + 12.0,
        0.0,
        100.0,
    )

    poses.append(
        (
            "7. THUMB YAW",
            THUMB_YAW,
        )
    )

    return poses


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        " L20Lite RIGHT Sim-Real Pose Comparison"
    )

    print(
        "=" * 72
    )

    sim = L20LiteMuJoCo(
        MODEL_PATH
    )

    with L20lite(
        side=REAL_SIDE,
        interface_name=CAN_INTERFACE,
    ) as hand:

        # ----------------------------------------------------
        # Read actual starting pose
        # ----------------------------------------------------

        data = (
            hand.angle.get_blocking(
                timeout_ms=1000,
            )
        )

        q0 = np.asarray(
            data.angles.to_list(),
            dtype=np.float64,
        )

        print()
        print(
            "Initial REAL pose:"
        )

        print(
            np.round(
                q0,
                2,
            ).tolist()
        )

        print()

        print(
            "Initial mapped SIM pose:"
        )

        print(
            np.round(
                real_sdk_to_mujoco(
                    q0
                ),
                2,
            ).tolist()
        )

        poses = build_poses(
            q0
        )

        # ----------------------------------------------------
        # Initialize simulation
        # ----------------------------------------------------

        sim.left.angle.set_angles(
            [0.0] * 10
        )

        sim.right.angle.set_angles(
            real_sdk_to_mujoco(
                q0
            )
        )

        for _ in range(
            800
        ):
            sim.step()

        # ----------------------------------------------------
        # Viewer
        # ----------------------------------------------------

        with mujoco.viewer.launch_passive(
            sim.model,
            sim.data,
        ) as viewer:

            viewer.cam.lookat[:] = (
                np.array(
                    [
                        0.0,
                        -0.18,
                        0.105,
                    ]
                )
            )

            viewer.cam.distance = (
                0.30
            )

            viewer.cam.azimuth = (
                90
            )

            viewer.cam.elevation = (
                -18
            )

            viewer.sync()

            current = q0.copy()

            print()
            print(
                "For every pose:"
            )

            print(
                "  REAL and MuJoCo move together."
            )

            print(
                "  Compare their final posture visually."
            )

            print()

            # ------------------------------------------------
            # Pose sequence
            # ------------------------------------------------

            try:

                for (
                    pose_name,
                    target,
                ) in poses:

                    if not viewer.is_running():
                        break

                    print_pose(
                        pose_name,
                        target,
                    )

                    input(
                        "\nPress ENTER to execute "
                        "this pose on REAL + SIM..."
                    )

                    move_pose(
                        hand,
                        sim,
                        viewer,
                        current,
                        target,
                    )

                    current = (
                        target.copy()
                    )

                    print()
                    print(
                        ">>> HOLDING POSE."
                    )

                    print(
                        "Compare REAL vs MuJoCo now."
                    )

                    input(
                        "Press ENTER to return "
                        "both to baseline..."
                    )

                    move_pose(
                        hand,
                        sim,
                        viewer,
                        current,
                        q0,
                    )

                    current = (
                        q0.copy()
                    )

                    print(
                        "[RETURNED TO BASE]"
                    )

                print()
                print(
                    "=" * 72
                )

                print(
                    "Pose comparison complete."
                )

                print(
                    "=" * 72
                )

                input(
                    "\nPress ENTER to close..."
                )

            except KeyboardInterrupt:

                print()
                print(
                    "[CTRL+C] Returning REAL + SIM "
                    "to initial pose..."
                )

                move_pose(
                    hand,
                    sim,
                    viewer,
                    current,
                    q0,
                    duration=1.0,
                )

                print(
                    "[RESTORED]"
                )


if __name__ == "__main__":
    main()
