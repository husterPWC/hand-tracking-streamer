#!/usr/bin/env python3

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
    REAL_SDK_ORDER,
    MUJOCO_JOINT_ORDER,
    real_sdk_to_mujoco,
)


MODEL_PATH = PROJECT_ROOT / "generated" / "dual_l20lite_dynamic.xml"

# ============================================================
# Test amount in REAL SDK space
# ============================================================

# Direction chosen so each motion is easy to see.
#
# flex: decrease -> bend
# abd : increase -> spread / semantic positive
# thumb yaw: increase
TEST_DELTA = np.array(
    [
        -10.0,   # 0 thumb_flex
        +10.0,   # 1 thumb_abd
        -10.0,   # 2 index_flex
        -10.0,   # 3 middle_flex
        -10.0,   # 4 ring_flex
        -10.0,   # 5 pinky_flex
        +10.0,   # 6 index_abd
        +10.0,   # 7 ring_abd
        +10.0,   # 8 pinky_abd
        +10.0,   # 9 thumb_yaw
    ],
    dtype=np.float64,
)

MOVE_TIME = 1.0
STEP_TIME = 0.05


def send_both(
    real_hand,
    sim,
    q_real,
):
    """
    Send one REAL-SDK-space command to:
      1. physical hand
      2. mapped MuJoCo hand
    """

    q_real = np.clip(
        np.asarray(q_real, dtype=np.float64),
        0.0,
        100.0,
    )

    q_sim = real_sdk_to_mujoco(
        q_real
    )

    # -----------------------------
    # REAL
    # -----------------------------
    real_hand.angle.set_angles(
        q_real.tolist()
    )

    # -----------------------------
    # SIM
    # -----------------------------
    sim.right.angle.set_angles(
        q_sim
    )

    sim.left.angle.set_angles(
        [0.0] * 10
    )


def move_both(
    real_hand,
    sim,
    viewer,
    q_start,
    q_target,
    duration=MOVE_TIME,
):
    """
    Smooth interpolation in REAL SDK coordinates.
    Same trajectory is mapped into MuJoCo.
    """

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
        int(duration / STEP_TIME),
    )

    for k in range(steps + 1):

        if not viewer.is_running():
            return

        a = k / steps

        # smoothstep
        s = a * a * (3.0 - 2.0 * a)

        q = (
            q_start
            + s * (q_target - q_start)
        )

        send_both(
            real_hand,
            sim,
            q,
        )

        # Let MuJoCo evolve during the same wall-clock period.
        n_sim_steps = max(
            1,
            int(STEP_TIME / sim.timestep),
        )

        for _ in range(n_sim_steps):
            sim.step()

        viewer.sync()

        time.sleep(STEP_TIME)


def main():

    print("=" * 72)
    print(" L20Lite RIGHT: REAL vs MuJoCo synchronized motion comparison")
    print("=" * 72)

    sim = L20LiteMuJoCo(
        MODEL_PATH
    )

    with L20lite(
        side="right",
        interface_name="can0",
    ) as real_hand:

        # ====================================================
        # Read current real hand as common baseline
        # ====================================================

        data = real_hand.angle.get_blocking(
            timeout_ms=1000
        )

        q0 = np.asarray(
            data.angles.to_list(),
            dtype=np.float64,
        )

        print("\nReal baseline:")
        print(
            np.round(q0, 2).tolist()
        )

        print("\nMapped MuJoCo baseline:")
        print(
            np.round(
                real_sdk_to_mujoco(q0),
                2,
            ).tolist()
        )

        # ====================================================
        # Initialize MuJoCo to same baseline
        # ====================================================

        sim.right.angle.set_angles(
            real_sdk_to_mujoco(q0)
        )

        sim.left.angle.set_angles(
            [0.0] * 10
        )

        for _ in range(800):
            sim.step()

        # ====================================================
        # Viewer
        # ====================================================

        with mujoco.viewer.launch_passive(
            sim.model,
            sim.data,
        ) as viewer:

            viewer.cam.lookat[:] = np.array(
                [
                    0.0,
                    -0.18,
                    0.105,
                ]
            )

            viewer.cam.distance = 0.30
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -18

            viewer.sync()

            print()
            print("The REAL hand and MuJoCo RIGHT hand")
            print("will now execute the SAME semantic motion.")
            print()

            # ================================================
            # Test each of 10 channels
            # ================================================

            for i in range(10):

                if not viewer.is_running():
                    break

                delta = TEST_DELTA[i]

                q_test = q0.copy()

                q_test[i] = np.clip(
                    q0[i] + delta,
                    0.0,
                    100.0,
                )

                print()
                print("=" * 72)

                print(
                    f"TEST [{i}] "
                    f"{REAL_SDK_ORDER[i]}"
                )

                print(
                    f"REAL: "
                    f"{q0[i]:.2f}"
                    f" -> "
                    f"{q_test[i]:.2f}"
                )

                qsim0 = (
                    real_sdk_to_mujoco(q0)
                )

                qsim1 = (
                    real_sdk_to_mujoco(q_test)
                )

                print(
                    f"SIM : "
                    f"{MUJOCO_JOINT_ORDER[i]}"
                    f" "
                    f"{qsim0[i]:.2f}"
                    f" -> "
                    f"{qsim1[i]:.2f}"
                )

                print("=" * 72)

                input(
                    "Press ENTER: move REAL + SIM together..."
                )

                move_both(
                    real_hand,
                    sim,
                    viewer,
                    q0,
                    q_test,
                )

                print()
                print(
                    ">>> LOOK AT BOTH HANDS NOW."
                )

                print(
                    "    Is the physical motion the SAME?"
                )

                input(
                    "Press ENTER: both return to baseline..."
                )

                move_both(
                    real_hand,
                    sim,
                    viewer,
                    q_test,
                    q0,
                )

                print(
                    "[RESTORED]"
                )

            print()
            print("=" * 72)
            print("10-D comparison completed.")
            print("=" * 72)

            input(
                "Press ENTER to close..."
            )


if __name__ == "__main__":
    main()
