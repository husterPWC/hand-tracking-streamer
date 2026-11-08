#!/usr/bin/env python3

import time
from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np


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


MODEL_PATH = PROJECT_ROOT / "generated" / "dual_l20lite_dynamic.xml"

JOINT_NAME = "thumb_cmc_roll"
CHANNEL_ID = 9

# 0 -> 80 corresponds to about:
# 0 -> 0.824 rad for the current URDF range [0, 1.03].
TARGET_VALUE = 80.0


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def move_right(sim, viewer, target_value, duration=1.2):
    start = np.asarray(
        sim.right.get_command(),
        dtype=np.float64,
    )

    target = np.zeros(10, dtype=np.float64)
    target[CHANNEL_ID] = target_value

    t0 = sim.data.time
    count = 0

    while viewer.is_running():
        alpha = (
            sim.data.time - t0
        ) / duration

        s = smoothstep(alpha)

        cmd = start + s * (target - start)

        # Right hand only.
        sim.right.angle.set_angles(cmd)
        sim.left.angle.set_angles([0.0] * 10)

        wall_t0 = time.perf_counter()

        sim.step()

        count += 1

        if count % 10 == 0:
            viewer.sync()

        sleep_time = (
            sim.timestep
            - (time.perf_counter() - wall_t0)
        )

        if sleep_time > 0:
            time.sleep(sleep_time)

        if alpha >= 1.0:
            break


def settle(sim, viewer, seconds=0.4):
    end_time = sim.data.time + seconds
    count = 0

    while (
        viewer.is_running()
        and sim.data.time < end_time
    ):
        wall_t0 = time.perf_counter()

        sim.step()

        count += 1

        if count % 10 == 0:
            viewer.sync()

        sleep_time = (
            sim.timestep
            - (time.perf_counter() - wall_t0)
        )

        if sleep_time > 0:
            time.sleep(sleep_time)


def print_state(sim, label):
    q = sim.right.joint_position(
        JOINT_NAME
    )

    actual = sim.right.angle.get_angles()

    print()
    print(f"[{label}]")
    print(
        f"thumb_cmc_roll = "
        f"{q:.5f} rad"
    )
    print(
        f"normalized     = "
        f"{actual[CHANNEL_ID]:.2f}"
    )


def main():
    sim = L20LiteMuJoCo(MODEL_PATH)

    zero = np.zeros(
        10,
        dtype=np.float64,
    )

    sim.left.angle.set_angles(zero)
    sim.right.angle.set_angles(zero)

    for _ in range(300):
        sim.step()

    print("=" * 72)
    print("L20Lite RIGHT thumb_cmc_roll dedicated probe")
    print("=" * 72)
    print()
    print("Only this joint will move:")
    print("  right_thumb_cmc_roll")
    print()
    print(
        f"Normalized command: "
        f"0 -> {TARGET_VALUE:.1f} -> 0"
    )
    print()
    print("IMPORTANT:")
    print("Do NOT only watch fingertip translation.")
    print("Watch the thumb surface / pad orientation.")
    print()
    print("Question:")
    print("Does the thumb rotate around its own longitudinal direction?")
    print()

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
    ) as viewer:

        # Focus camera tightly on RIGHT thumb.
        #
        # Right thumb base in current MJCF is approximately:
        # (-0.013, -0.167, 0.061)
        #
        # Look slightly higher toward the whole thumb.
        viewer.cam.lookat[:] = np.array(
            [
                0.005,
                -0.165,
                0.095,
            ],
            dtype=np.float64,
        )

        viewer.cam.distance = 0.20

        # Initial close-up view.
        # You can still drag the mouse in the viewer
        # if another angle makes the roll easier to see.
        viewer.cam.azimuth = 35
        viewer.cam.elevation = -10

        viewer.sync()

        print_state(
            sim,
            "ZERO",
        )

        input(
            "\nPress ENTER: "
            "thumb_cmc_roll 0 -> 80..."
        )

        move_right(
            sim,
            viewer,
            TARGET_VALUE,
            duration=1.2,
        )

        settle(
            sim,
            viewer,
            seconds=0.5,
        )

        print_state(
            sim,
            "POSITIVE",
        )

        print()
        print("=" * 72)
        print("NOW OBSERVE")
        print("=" * 72)
        print()
        print("1. Does the fingertip move toward/away from index?")
        print("2. More importantly: does the thumb PAD/Nail orientation rotate?")
        print("3. If yes, which way does the pad turn?")
        print()

        input(
            "Press ENTER: restore 80 -> 0..."
        )

        move_right(
            sim,
            viewer,
            0.0,
            duration=1.2,
        )

        settle(
            sim,
            viewer,
            seconds=0.5,
        )

        print_state(
            sim,
            "RESTORED",
        )

        print()
        print("Test finished.")
        print(
            "You can rotate the MuJoCo camera with the mouse "
            "and run the script again if needed."
        )

        input(
            "\nPress ENTER to close..."
        )


if __name__ == "__main__":
    main()
