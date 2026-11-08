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


from l20lite_sim import L20LiteMuJoCo, SDK_JOINT_ORDER


MODEL_PATH = PROJECT_ROOT / "generated" / "dual_l20lite_dynamic.xml"

# MuJoCo normalized command.
# 0 -> joint lower limit
# TEST_VALUE -> positive joint direction
TEST_VALUE = 40.0

# Flex can use a slightly larger value for easy observation.
TEST_VALUES = {
    0: 40.0,   # thumb_cmc_pitch
    1: 35.0,   # thumb_cmc_yaw
    2: 40.0,   # index_mcp_pitch
    3: 40.0,   # middle_mcp_pitch
    4: 40.0,   # ring_mcp_pitch
    5: 40.0,   # pinky_mcp_pitch
    6: 60.0,   # index_mcp_roll
    7: 60.0,   # ring_mcp_roll
    8: 60.0,   # pinky_mcp_roll
    9: 40.0,   # thumb_cmc_roll
}


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def move_right(sim, viewer, target, duration=0.8):
    start = np.asarray(
        sim.right.get_command(),
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    t0 = sim.data.time
    count = 0

    while viewer.is_running():
        alpha = (sim.data.time - t0) / duration
        s = smoothstep(alpha)

        cmd = start + s * (target - start)

        sim.right.angle.set_angles(cmd)
        sim.left.angle.set_angles([0.0] * 10)

        wall_t0 = time.perf_counter()
        sim.step()

        count += 1
        if count % 10 == 0:
            viewer.sync()

        sleep_time = sim.timestep - (
            time.perf_counter() - wall_t0
        )

        if sleep_time > 0:
            time.sleep(sleep_time)

        if alpha >= 1.0:
            break


def hold(sim, viewer, seconds=0.3):
    end = sim.data.time + seconds
    count = 0

    while viewer.is_running() and sim.data.time < end:
        t0 = time.perf_counter()

        sim.step()

        count += 1
        if count % 10 == 0:
            viewer.sync()

        dt = sim.timestep - (
            time.perf_counter() - t0
        )

        if dt > 0:
            time.sleep(dt)


def main():
    sim = L20LiteMuJoCo(MODEL_PATH)

    print("=" * 72)
    print("L20Lite MuJoCo RIGHT 10-D positive-direction probe")
    print("=" * 72)
    print()
    print("Meaning of this test:")
    print("  command 0 -> positive normalized value")
    print("  therefore joint angle goes toward its POSITIVE URDF direction.")
    print()
    print("We only move the RIGHT hand.")
    print()

    for i, name in enumerate(SDK_JOINT_ORDER):
        print(f"[{i}] {name}")

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
    ) as viewer:

        # Focus more toward right hand.
        viewer.cam.lookat[:] = np.array(
            [0.0, -0.18, 0.10]
        )

        viewer.cam.distance = 0.32
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -18

        viewer.sync()

        zero = np.zeros(10, dtype=np.float64)

        # Ensure open / lower-limit pose.
        sim.right.angle.set_angles(zero)
        sim.left.angle.set_angles(zero)

        for _ in range(300):
            sim.step()

        viewer.sync()

        for joint_id, joint_name in enumerate(
            SDK_JOINT_ORDER
        ):
            print()
            print("=" * 72)
            print(
                f"TEST [{joint_id}] {joint_name}"
            )
            print("=" * 72)

            value = TEST_VALUES[joint_id]

            print(
                f"MuJoCo normalized command: "
                f"0 -> {value:.1f}"
            )

            input(
                "Press ENTER to move in POSITIVE joint direction..."
            )

            target = np.zeros(
                10,
                dtype=np.float64,
            )
            target[joint_id] = value

            move_right(
                sim,
                viewer,
                target,
                duration=0.8,
            )

            hold(
                sim,
                viewer,
                seconds=0.4,
            )

            q = sim.right.joint_position(
                joint_name
            )

            actual = (
                sim.right.angle.get_angles()
            )

            print(
                f"\nJoint position = {q:.5f} rad"
            )
            print(
                f"Normalized actual = "
                f"{actual[joint_id]:.2f}"
            )

            print()
            print("OBSERVE THE MOTION DIRECTION.")

            input(
                "Press ENTER to restore to 0..."
            )

            move_right(
                sim,
                viewer,
                zero,
                duration=0.8,
            )

            hold(
                sim,
                viewer,
                seconds=0.3,
            )

            print("[RESTORED]")

        print()
        print("=" * 72)
        print("All 10 positive-direction tests completed.")
        print("=" * 72)

        input("Press ENTER to close viewer...")


if __name__ == "__main__":
    main()
