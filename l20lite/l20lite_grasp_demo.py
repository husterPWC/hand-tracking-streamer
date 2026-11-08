#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from l20lite_sim import L20LiteMuJoCo


ROOT = Path(__file__).resolve().parent

# ============================================================
# IMPORTANT
#
# This stage validates ONLY the natural hand motion.
# No object, no contact, no weld.
#
# Use the stable free-space dynamic model.
# ============================================================

MODEL_PATH = (
    ROOT
    / "generated"
    / "dual_l20lite_dynamic.xml"
)


# ============================================================
# L20 Lite control order
#
# 0 : thumb_cmc_pitch
# 1 : thumb_cmc_yaw
# 2 : index_mcp_pitch
# 3 : middle_mcp_pitch
# 4 : ring_mcp_pitch
# 5 : pinky_mcp_pitch
# 6 : index_mcp_roll
# 7 : ring_mcp_roll
# 8 : pinky_mcp_roll
# 9 : thumb_cmc_roll
#
# Based on our manual observation:
#
# thumb pitch [0]:
#     thumb moves toward index
#     + fingertip bends
#
# thumb yaw [1]:
#     thumb moves toward palm
#
# thumb roll [9]:
#     thumb moves laterally toward index,
#     but fingertip itself bends very little
# ============================================================


# ============================================================
# Stage 0: natural open hand
# ============================================================

HAND_OPEN = np.array(
    [
        0,      # thumb pitch
        0,      # thumb yaw

        4,      # index slight relaxation

        6,      # middle
        8,      # ring
        10,     # pinky

        0,      # index roll
        0,      # ring roll
        0,      # pinky roll

        0,      # thumb roll
    ],
    dtype=np.float64,
)


# ============================================================
# Stage 1: thumb opposition
#
# Main motion:
#     thumb roll ↑
#
# Small amount:
#     thumb yaw ↑
#
# Almost no thumb pitch yet.
#
# Goal:
#     move the whole thumb toward the index side
#     WITHOUT curling the thumb tip too early.
# ============================================================

THUMB_OPPOSITION = np.array(
    [
        3,      # thumb pitch：暂时少动
        18,     # thumb yaw：10 -> 18，向掌心转

        7,

        8,
        10,
        13,

        0,
        0,
        0,

        24,     # thumb roll：先保持不变
    ],
    dtype=np.float64,
)


# ============================================================
# Stage 2: pinch preshape
#
# Thumb is already on the index side.
#
# Now:
#     slightly more roll
#     more yaw toward palm
#     slight pitch
#     index begins to flex
#
# Goal:
#
#       index
#          \
#           \
#            space
#           /
#       thumb
#
# but still NO full closure.
# ============================================================

PINCH_PRESHAPE = np.array(
    [
        8,
        30,     # 18 -> 30，明显进入掌前工作空间

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


# ============================================================
# Stage 3: precision pinch closure
#
# Main closure DOFs:
#
#     thumb pitch ↑↑
#     index flex  ↑↑
#
# Thumb roll only increases slightly.
# Thumb yaw only increases modestly.
#
# This is the critical difference from our old trajectory.
# ============================================================

PINCH_CLOSE = np.array(
    [
        45,     # pitch：负责最后弯向食指
        36,     # yaw：25 -> 36，保持对掌位置

        50,     # index flex

        14,
        17,
        20,

        0,
        0,
        0,

        38,     # roll 暂时不再增加
    ],
    dtype=np.float64,
)


# ============================================================
# Optional slightly stronger pinch
#
# Only for visual inspection after PINCH_CLOSE.
#
# Still not using extreme 100% values.
# ============================================================

PINCH_TIGHT = np.array(
    [
        55,     # thumb pitch
        28,     # thumb yaw

        58,     # index flex

        15,
        18,
        21,

        0,
        0,
        0,

        40,     # thumb roll
    ],
    dtype=np.float64,
)


# ============================================================
# Helpers
# ============================================================

def smoothstep(x: float) -> float:

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


def interpolate_command(
    start_cmd,
    target_cmd,
    alpha,
):

    s = smoothstep(
        alpha
    )

    return (
        start_cmd
        + s
        * (
            target_cmd
            - start_cmd
        )
    )


def realtime_step(
    sim,
):

    wall_start = (
        time.perf_counter()
    )

    sim.step()

    elapsed = (
        time.perf_counter()
        - wall_start
    )

    remaining = (
        sim.timestep
        - elapsed
    )

    if remaining > 0:

        time.sleep(
            remaining
        )


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


def move_to(
    sim,
    viewer,
    start_cmd,
    target_cmd,
    duration,
    label,
):

    print(
        f"\n[MOVE] {label}"
    )

    start_time = (
        sim.data.time
    )

    counter = 0

    while viewer.is_running():

        alpha = np.clip(
            (
                sim.data.time
                - start_time
            )
            / duration,
            0.0,
            1.0,
        )

        cmd = interpolate_command(
            start_cmd,
            target_cmd,
            alpha,
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

    actual = np.asarray(
        sim.right.angle.get_angles(),
        dtype=np.float64,
    )

    print(
        " target:",
        np.round(
            target_cmd,
            1,
        ).tolist()
    )

    print(
        " actual:",
        np.round(
            actual,
            1,
        ).tolist()
    )

    print(
        " thumb:"
        f" pitch={actual[0]:.1f},"
        f" yaw={actual[1]:.1f},"
        f" roll={actual[9]:.1f}"
    )

    print(
        f" index flex={actual[2]:.1f}"
    )


# ============================================================
# Optional joint-angle print
# ============================================================

def print_thumb_joint_radians(
    sim,
):

    pitch = sim.right.joint_position(
        "thumb_cmc_pitch"
    )

    yaw = sim.right.joint_position(
        "thumb_cmc_yaw"
    )

    roll = sim.right.joint_position(
        "thumb_cmc_roll"
    )

    index = sim.right.joint_position(
        "index_mcp_pitch"
    )

    print(
        "\n[JOINT RAD]"
    )

    print(
        f" thumb pitch = "
        f"{pitch:.3f} rad"
    )

    print(
        f" thumb yaw   = "
        f"{yaw:.3f} rad"
    )

    print(
        f" thumb roll  = "
        f"{roll:.3f} rad"
    )

    print(
        f" index flex  = "
        f"{index:.3f} rad"
    )


# ============================================================
# Single natural pinch sequence
# ============================================================

def run_sequence(
    sim,
    viewer,
    include_tight,
):

    # --------------------------------------------------------
    # 0. OPEN
    # --------------------------------------------------------

    move_to(
        sim,
        viewer,
        np.asarray(
            sim.right.get_command(),
            dtype=np.float64,
        ),
        HAND_OPEN,
        duration=1.0,
        label="NATURAL OPEN",
    )

    hold(
        sim,
        viewer,
        0.8,
    )

    # --------------------------------------------------------
    # 1. THUMB OPPOSITION
    #
    # Primarily thumb roll.
    # --------------------------------------------------------

    move_to(
        sim,
        viewer,
        HAND_OPEN,
        THUMB_OPPOSITION,
        duration=1.2,
        label="THUMB OPPOSITION",
    )

    hold(
        sim,
        viewer,
        0.8,
    )

    # --------------------------------------------------------
    # 2. PINCH PRESHAPE
    #
    # Thumb moves slightly into palm,
    # index begins natural flexion.
    # --------------------------------------------------------

    move_to(
        sim,
        viewer,
        THUMB_OPPOSITION,
        PINCH_PRESHAPE,
        duration=1.2,
        label="PINCH PRESHAPE",
    )

    hold(
        sim,
        viewer,
        0.8,
    )

    # --------------------------------------------------------
    # 3. COORDINATED CLOSE
    #
    # Main:
    #     thumb pitch
    #     index flex
    # --------------------------------------------------------

    move_to(
        sim,
        viewer,
        PINCH_PRESHAPE,
        PINCH_CLOSE,
        duration=1.5,
        label="PRECISION PINCH CLOSE",
    )

    hold(
        sim,
        viewer,
        1.2,
    )

    print_thumb_joint_radians(
        sim
    )

    # --------------------------------------------------------
    # Optional extra closure
    # --------------------------------------------------------

    if include_tight:

        move_to(
            sim,
            viewer,
            PINCH_CLOSE,
            PINCH_TIGHT,
            duration=0.7,
            label="SLIGHTLY TIGHTER PINCH",
        )

        hold(
            sim,
            viewer,
            1.0,
        )

    # --------------------------------------------------------
    # Return to open
    # --------------------------------------------------------

    current = np.asarray(
        sim.right.get_command(),
        dtype=np.float64,
    )

    move_to(
        sim,
        viewer,
        current,
        PINCH_PRESHAPE,
        duration=0.8,
        label="RELEASE TO PRESHAPE",
    )

    move_to(
        sim,
        viewer,
        PINCH_PRESHAPE,
        THUMB_OPPOSITION,
        duration=0.8,
        label="RELEASE OPPOSITION",
    )

    move_to(
        sim,
        viewer,
        THUMB_OPPOSITION,
        HAND_OPEN,
        duration=1.0,
        label="RETURN OPEN",
    )

    hold(
        sim,
        viewer,
        1.0,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Run one pinch cycle and "
            "keep the viewer open."
        ),
    )

    parser.add_argument(
        "--tight",
        action="store_true",
        help=(
            "Also show the slightly tighter "
            "pinch posture."
        ),
    )

    args = parser.parse_args()

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"{MODEL_PATH}\n"
            "Run build_l20lite_dynamic.py first."
        )

    sim = L20LiteMuJoCo(
        MODEL_PATH
    )

    print(
        "========================================"
    )

    print(
        " L20 Lite Natural Pinch Motion Test"
    )

    print(
        "========================================"
    )

    print(
        f"model:"
        f" {MODEL_PATH}"
    )

    print(
        f"nq={sim.model.nq}, "
        f"nv={sim.model.nv}, "
        f"nu={sim.model.nu}, "
        f"neq={sim.model.neq}"
    )

    print(
        "\nManual thumb interpretation:"
    )

    print(
        "  pitch [0] : toward index "
        "+ fingertip flex"
    )

    print(
        "  yaw   [1] : toward palm"
    )

    print(
        "  roll  [9] : lateral toward index"
    )

    print(
        "\nMotion sequence:"
    )

    print(
        "  OPEN"
    )

    print(
        "    -> THUMB OPPOSITION"
    )

    print(
        "    -> PINCH PRESHAPE"
    )

    print(
        "    -> PRECISION PINCH CLOSE"
    )

    if args.tight:

        print(
            "    -> SLIGHTLY TIGHTER PINCH"
        )

    print(
        "    -> OPEN"
    )

    # Left hand simply stays open.
    sim.left.angle.set_angles(
        [0] * 10
    )

    sim.right.angle.set_angles(
        HAND_OPEN
    )

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
    ) as viewer:

        # Focus on the RIGHT hand.
        viewer.cam.lookat[:] = np.array(
            [
                0.02,
                -0.18,
                0.14,
            ],
            dtype=np.float64,
        )

        viewer.cam.distance = (
            0.27
        )

        # Front/palm-biased view.
        viewer.cam.azimuth = (
            90
        )

        viewer.cam.elevation = (
            -8
        )

        viewer.sync()

        # Let initial state settle.
        hold(
            sim,
            viewer,
            0.8,
        )

        if args.once:

            run_sequence(
                sim,
                viewer,
                include_tight=args.tight,
            )

            print(
                "\n[DONE]"
            )

            print(
                "Pinch motion sequence completed."
            )

            print(
                "Close the viewer when finished inspecting."
            )

            while viewer.is_running():

                sim.right.angle.set_angles(
                    HAND_OPEN
                )

                realtime_step(
                    sim
                )

                viewer.sync()

            return

        # Repeated cycle
        while viewer.is_running():

            run_sequence(
                sim,
                viewer,
                include_tight=args.tight,
            )


if __name__ == "__main__":
    main()
