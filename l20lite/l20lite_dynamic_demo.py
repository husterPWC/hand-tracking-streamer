#!/usr/bin/env python3

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
    / "dual_l20lite_dynamic.xml"
)


# ============================================================
# L20 Lite commands
# ============================================================

OPEN = np.array(
    [
        0,      # thumb flex
        0,      # thumb yaw
        0,      # index flex
        0,      # middle flex
        0,      # ring flex
        0,      # pinky flex
        0,      # index side
        0,      # ring side
        0,      # pinky side
        0,      # thumb roll
    ],
    dtype=np.float64,
)


FIST = np.array(
    [
        95,
        55,
        100,
        100,
        100,
        100,
        0,
        0,
        0,
        60,
    ],
    dtype=np.float64,
)


HALF = np.array(
    [
        50,
        30,
        55,
        55,
        55,
        55,
        0,
        0,
        0,
        35,
    ],
    dtype=np.float64,
)


THUMB_OPPOSE = np.array(
    [
        70,
        60,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        80,
    ],
    dtype=np.float64,
)


INDEX_ONLY = np.array(
    [
        0,
        0,
        100,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    dtype=np.float64,
)


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


# ============================================================

def move(
    sim,
    viewer,
    left_target,
    right_target,
    duration=1.2,
):

    left_start = np.asarray(
        sim.left.get_command(),
        dtype=np.float64,
    )

    right_start = np.asarray(
        sim.right.get_command(),
        dtype=np.float64,
    )

    start_sim_time = (
        sim.data.time
    )

    step_count = 0

    while viewer.is_running():

        elapsed = (
            sim.data.time
            - start_sim_time
        )

        alpha = (
            elapsed / duration
        )

        s = smoothstep(
            alpha
        )

        left_cmd = (
            left_start
            + s
            * (
                left_target
                - left_start
            )
        )

        right_cmd = (
            right_start
            + s
            * (
                right_target
                - right_start
            )
        )

        sim.left.angle.set_angles(
            left_cmd
        )

        sim.right.angle.set_angles(
            right_cmd
        )

        wall_start = (
            time.perf_counter()
        )

        sim.step()

        step_count += 1

        # Viewer roughly 50 Hz.
        if step_count % 10 == 0:
            viewer.sync()

        # approximately real-time
        elapsed_wall = (
            time.perf_counter()
            - wall_start
        )

        sleep_time = (
            sim.timestep
            - elapsed_wall
        )

        if sleep_time > 0:
            time.sleep(
                sleep_time
            )

        if alpha >= 1.0:
            break


# ============================================================

def hold(
    sim,
    viewer,
    duration=0.5,
):

    end_time = (
        sim.data.time
        + duration
    )

    step_count = 0

    while (
        viewer.is_running()
        and sim.data.time < end_time
    ):

        wall_start = (
            time.perf_counter()
        )

        sim.step()

        step_count += 1

        if step_count % 10 == 0:
            viewer.sync()

        elapsed_wall = (
            time.perf_counter()
            - wall_start
        )

        sleep_time = (
            sim.timestep
            - elapsed_wall
        )

        if sleep_time > 0:
            time.sleep(
                sleep_time
            )


# ============================================================

def print_status(
    sim,
    label,
):

    left = np.asarray(
        sim.left.angle.get_angles()
    )

    right = np.asarray(
        sim.right.angle.get_angles()
    )

    print(
        f"\n[{label}]"
    )

    print(
        " left actual :",
        np.round(
            left,
            1,
        ).tolist()
    )

    print(
        " right actual:",
        np.round(
            right,
            1,
        ).tolist()
    )

    # Verify mechanical coupling.
    q_mcp = (
        sim.left
        .joint_position(
            "index_mcp_pitch"
        )
    )

    q_pip = (
        sim.left
        .joint_position(
            "index_pip"
        )
    )

    q_dip = (
        sim.left
        .joint_position(
            "index_dip"
        )
    )

    print(
        " left index(rad): "
        f"MCP={q_mcp:.3f}, "
        f"PIP={q_pip:.3f}, "
        f"DIP={q_dip:.3f}"
    )

    if abs(q_mcp) > 0.01:

        print(
            " coupling ratio: "
            f"PIP/MCP={q_pip/q_mcp:.3f}, "
            f"DIP/MCP={q_dip/q_mcp:.3f}"
        )


# ============================================================

def main():

    sim = L20LiteMuJoCo(
        MODEL_PATH
    )

    print(
        "================================="
    )

    print(
        " L20 Lite Dynamic Simulation"
    )

    print(
        "================================="
    )

    print(
        f"model: {MODEL_PATH}"
    )

    print(
        f"timestep: "
        f"{sim.timestep:.4f} s"
    )

    print(
        f"nq={sim.model.nq}, "
        f"nv={sim.model.nv}, "
        f"nu={sim.model.nu}, "
        f"neq={sim.model.neq}"
    )

    print(
        "\nControl interface:"
    )

    print(
        "  sim.left.angle.set_angles("
        "[10 values])"
    )

    print(
        "  sim.right.angle.set_angles("
        "[10 values])"
    )

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
    ) as viewer:

        viewer.cam.lookat[:] = (
            np.array(
                [
                    0.0,
                    0.0,
                    0.10,
                ]
            )
        )

        viewer.cam.distance = 0.48
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -18

        viewer.sync()

        sequence = [

            (
                OPEN,
                OPEN,
                "BOTH OPEN",
            ),

            (
                FIST,
                FIST,
                "BOTH FIST",
            ),

            (
                OPEN,
                FIST,
                "LEFT OPEN / RIGHT FIST",
            ),

            (
                FIST,
                OPEN,
                "LEFT FIST / RIGHT OPEN",
            ),

            (
                INDEX_ONLY,
                INDEX_ONLY,
                "INDEX ONLY",
            ),

            (
                THUMB_OPPOSE,
                THUMB_OPPOSE,
                "THUMB OPPOSE",
            ),

            (
                HALF,
                HALF,
                "HALF CLOSE",
            ),

            (
                OPEN,
                OPEN,
                "BOTH OPEN",
            ),
        ]

        while viewer.is_running():

            for (
                left_target,
                right_target,
                label,
            ) in sequence:

                if not viewer.is_running():
                    break

                print(
                    f"\n[MOVE] {label}"
                )

                move(
                    sim,
                    viewer,
                    left_target,
                    right_target,
                    duration=1.2,
                )

                hold(
                    sim,
                    viewer,
                    duration=0.6,
                )

                print_status(
                    sim,
                    label,
                )


if __name__ == "__main__":
    main()
