#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from l20lite_sim import L20LiteMuJoCo


ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "generated"
    / "dual_l20lite_tactile.xml"
)


# ============================================================
# Use the natural pinch posture we just tuned.
#
# Since you have already increased thumb yaw and said the
# posture now looks much better, keep this as the current
# natural test trajectory.
# ============================================================

HAND_OPEN = np.array(
    [
        0,
        0,

        4,

        6,
        8,
        10,

        0,
        0,
        0,

        0,
    ],
    dtype=np.float64,
)


THUMB_OPPOSITION = np.array(
    [
        3,

        18,     # more inward thumb yaw

        7,

        8,
        10,
        13,

        0,
        0,
        0,

        24,
    ],
    dtype=np.float64,
)


PINCH_PRESHAPE = np.array(
    [
        8,

        30,     # tuned inward yaw

        13,

        11,
        14,
        17,

        0,
        0,
        0,

        32,
    ],
    dtype=np.float64,
)


PINCH_CLOSE = np.array(
    [
        45,

        36,

        50,

        14,
        17,
        20,

        0,
        0,
        0,

        38,
    ],
    dtype=np.float64,
)


def smoothstep(
    x,
):

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


def realtime_step(
    sim,
):

    start = (
        time.perf_counter()
    )

    sim.step()

    elapsed = (
        time.perf_counter()
        - start
    )

    wait = (
        sim.timestep
        - elapsed
    )

    if wait > 0:

        time.sleep(
            wait
        )


def move(
    sim,
    viewer,
    q0,
    q1,
    duration,
    label,
):

    print(
        f"\n[MOVE] {label}"
    )

    t0 = (
        sim.data.time
    )

    counter = 0

    while viewer.is_running():

        alpha = np.clip(
            (
                sim.data.time
                - t0
            )
            / duration,
            0.0,
            1.0,
        )

        s = smoothstep(
            alpha
        )

        cmd = (
            q0
            + s
            * (
                q1
                - q0
            )
        )

        sim.right.angle.set_angles(
            cmd
        )

        realtime_step(
            sim
        )

        counter += 1

        if counter % 10 == 0:

            viewer.sync()

        if alpha >= 1.0:

            break


def hold(
    sim,
    viewer,
    duration,
):

    end_time = (
        sim.data.time
        + duration
    )

    counter = 0

    while (
        viewer.is_running()
        and sim.data.time < end_time
    ):

        realtime_step(
            sim
        )

        counter += 1

        if counter % 10 == 0:

            viewer.sync()


def print_tip_positions(
    sim,
    label,
):

    print(
        f"\n[{label}]"
    )

    for finger in (
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    ):

        name = (
            f"right_{finger}_tip_site"
        )

        sid = mujoco.mj_name2id(
            sim.model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )

        pos = (
            sim.data.site_xpos[
                sid
            ]
        )

        print(
            f"  {finger:<7s}: "
            f"[{pos[0]:.4f}, "
            f"{pos[1]:.4f}, "
            f"{pos[2]:.4f}]"
        )

    thumb_sid = mujoco.mj_name2id(
        sim.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "right_thumb_tip_site",
    )

    index_sid = mujoco.mj_name2id(
        sim.model,
        mujoco.mjtObj.mjOBJ_SITE,
        "right_index_tip_site",
    )

    thumb = (
        sim.data.site_xpos[
            thumb_sid
        ]
    )

    index = (
        sim.data.site_xpos[
            index_sid
        ]
    )

    gap = float(
        np.linalg.norm(
            thumb
            - index
        )
    )

    midpoint = (
        0.5
        * (
            thumb
            + index
        )
    )

    print(
        f"\n  thumb-index gap:"
        f" {gap*1000:.1f} mm"
    )

    print(
        "  pinch midpoint:",
        np.round(
            midpoint,
            4,
        ).tolist()
    )


def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"{MODEL_PATH}\n"
            "Run build_l20lite_tactile.py first."
        )

    sim = L20LiteMuJoCo(
        MODEL_PATH
    )

    sim.left.angle.set_angles(
        [0] * 10
    )

    sim.right.angle.set_angles(
        HAND_OPEN
    )

    print(
        "========================================"
    )

    print(
        " L20 Lite Fingertip Site Test"
    )

    print(
        "========================================"
    )

    print(
        f"nsite={sim.model.nsite}"
    )

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
    ) as viewer:

        # Make sites visible.
        viewer.opt.sitegroup[:] = 1

        viewer.cam.lookat[:] = np.array(
            [
                0.02,
                -0.18,
                0.14,
            ],
            dtype=np.float64,
        )

        viewer.cam.distance = 0.27
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -8

        viewer.sync()

        hold(
            sim,
            viewer,
            0.6,
        )

        print_tip_positions(
            sim,
            "OPEN",
        )

        move(
            sim,
            viewer,
            HAND_OPEN,
            THUMB_OPPOSITION,
            1.2,
            "THUMB OPPOSITION",
        )

        hold(
            sim,
            viewer,
            0.6,
        )

        print_tip_positions(
            sim,
            "THUMB OPPOSITION",
        )

        move(
            sim,
            viewer,
            THUMB_OPPOSITION,
            PINCH_PRESHAPE,
            1.2,
            "PINCH PRESHAPE",
        )

        hold(
            sim,
            viewer,
            0.8,
        )

        print_tip_positions(
            sim,
            "PINCH PRESHAPE",
        )

        move(
            sim,
            viewer,
            PINCH_PRESHAPE,
            PINCH_CLOSE,
            1.5,
            "PINCH CLOSE",
        )

        hold(
            sim,
            viewer,
            1.5,
        )

        print_tip_positions(
            sim,
            "PINCH CLOSE",
        )

        print(
            "\n[DONE]"
        )

        print(
            "Red spheres should sit on "
            "the physical fingertips."
        )

        while viewer.is_running():

            sim.right.angle.set_angles(
                PINCH_CLOSE
            )

            realtime_step(
                sim
            )

            viewer.sync()


if __name__ == "__main__":
    main()
