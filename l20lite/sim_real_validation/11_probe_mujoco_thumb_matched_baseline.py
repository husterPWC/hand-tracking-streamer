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

# ------------------------------------------------------------
# Real-hand reference posture measured during our tests
# ------------------------------------------------------------

REAL_THUMB_FLEX = 99.22
REAL_THUMB_ABD  = 48.63
REAL_THUMB_YAW  = 43.92

# ------------------------------------------------------------
# H0: official/intended one-to-one mechanical correspondence
#
# real thumb_flex -> MuJoCo thumb_cmc_pitch
# real thumb_abd  -> MuJoCo thumb_cmc_yaw
# real thumb_yaw  -> MuJoCo thumb_cmc_roll
#
# Flex polarity already experimentally confirmed reversed.
# ABD / YAW polarity are intentionally NOT assumed here.
# ------------------------------------------------------------

BASE = np.zeros(10, dtype=np.float64)

BASE[0] = 100.0 - REAL_THUMB_FLEX
BASE[1] = REAL_THUMB_ABD
BASE[9] = REAL_THUMB_YAW

TESTS = [
    # channel, joint, delta
    (0, "thumb_cmc_pitch", +20.0),
    (1, "thumb_cmc_yaw",   +15.0),
    (9, "thumb_cmc_roll",  +15.0),
]

STEP_DURATION = 1.0


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def move(sim, viewer, target, duration=1.0):
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

        alpha = (
            sim.data.time - t0
        ) / duration

        s = smoothstep(alpha)

        cmd = start + s * (
            target - start
        )

        sim.right.angle.set_angles(cmd)
        sim.left.angle.set_angles([0.0] * 10)

        wall0 = time.perf_counter()

        sim.step()

        count += 1

        if count % 10 == 0:
            viewer.sync()

        sleep_time = (
            sim.timestep
            - (time.perf_counter() - wall0)
        )

        if sleep_time > 0:
            time.sleep(sleep_time)

        if alpha >= 1.0:
            break


def settle(sim, viewer, seconds=0.4):

    end = sim.data.time + seconds
    count = 0

    while (
        viewer.is_running()
        and sim.data.time < end
    ):

        t0 = time.perf_counter()

        sim.step()

        count += 1

        if count % 10 == 0:
            viewer.sync()

        sleep_time = (
            sim.timestep
            - (time.perf_counter() - t0)
        )

        if sleep_time > 0:
            time.sleep(sleep_time)


def main():

    sim = L20LiteMuJoCo(
        MODEL_PATH
    )

    print("=" * 72)
    print("L20Lite RIGHT thumb matched-baseline test")
    print("=" * 72)

    print()
    print("Real reference:")
    print(
        f"  thumb_flex = {REAL_THUMB_FLEX:.2f}"
    )
    print(
        f"  thumb_abd  = {REAL_THUMB_ABD:.2f}"
    )
    print(
        f"  thumb_yaw  = {REAL_THUMB_YAW:.2f}"
    )

    print()
    print("MuJoCo H0 reference:")
    print(
        f"  cmc_pitch = {BASE[0]:.2f}"
    )
    print(
        f"  cmc_yaw   = {BASE[1]:.2f}"
    )
    print(
        f"  cmc_roll  = {BASE[9]:.2f}"
    )

    # Start from reference.
    sim.right.angle.set_angles(BASE)
    sim.left.angle.set_angles([0.0] * 10)

    for _ in range(800):
        sim.step()

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
    ) as viewer:

        viewer.cam.lookat[:] = np.array(
            [
                0.005,
                -0.165,
                0.095,
            ]
        )

        viewer.cam.distance = 0.22
        viewer.cam.azimuth = 35
        viewer.cam.elevation = -10

        viewer.sync()

        print()
        input(
            "Inspect BASE posture, then press ENTER..."
        )

        for channel, name, delta in TESTS:

            print()
            print("=" * 72)
            print(
                f"TEST [{channel}] {name}"
            )
            print("=" * 72)

            target = BASE.copy()

            target[channel] = np.clip(
                target[channel] + delta,
                0.0,
                100.0,
            )

            print(
                f"{BASE[channel]:.2f}"
                f" -> "
                f"{target[channel]:.2f}"
            )

            input(
                "Press ENTER to apply positive perturbation..."
            )

            move(
                sim,
                viewer,
                target,
                STEP_DURATION,
            )

            settle(
                sim,
                viewer,
            )

            q = sim.right.joint_position(
                name
            )

            print(
                f"joint = {q:.5f} rad"
            )

            print()
            print("Observe:")
            print("  - position direction")
            print("  - thumb pad orientation")
            print("  - axial rotation if any")

            input(
                "Press ENTER to restore BASE..."
            )

            move(
                sim,
                viewer,
                BASE,
                STEP_DURATION,
            )

            settle(
                sim,
                viewer,
            )

        print()
        print("Finished.")

        input(
            "Press ENTER to close..."
        )


if __name__ == "__main__":
    main()
