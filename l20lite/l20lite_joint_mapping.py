#!/usr/bin/env python3
"""
L20Lite real-SDK <-> MuJoCo 10-D mapping.

RIGHT hand mapping experimentally verified on real hardware
and MuJoCo using matched thumb baseline tests.

Real SDK order:
    0 thumb_flex
    1 thumb_abd
    2 index_flex
    3 middle_flex
    4 ring_flex
    5 pinky_flex
    6 index_abd
    7 ring_abd
    8 pinky_abd
    9 thumb_yaw

MuJoCo active-joint order:
    0 thumb_cmc_pitch
    1 thumb_cmc_yaw
    2 index_mcp_pitch
    3 middle_mcp_pitch
    4 ring_mcp_pitch
    5 pinky_mcp_pitch
    6 index_mcp_roll
    7 ring_mcp_roll
    8 pinky_mcp_roll
    9 thumb_cmc_roll
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Semantic real-hand SDK order
# ============================================================

REAL_SDK_ORDER = (
    "thumb_flex",
    "thumb_abd",
    "index_flex",
    "middle_flex",
    "ring_flex",
    "pinky_flex",
    "index_abd",
    "ring_abd",
    "pinky_abd",
    "thumb_yaw",
)


# ============================================================
# MuJoCo active-joint order
# ============================================================

MUJOCO_JOINT_ORDER = (
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
# Experimentally verified polarity
#
# True:
#     real SDK increasing corresponds to
#     MuJoCo normalized command decreasing.
#
# False:
#     same direction.
# ============================================================

INVERT_MASK = np.array(
    [
        True,   # 0 thumb_flex
        True,   # 1 thumb_abd
        True,   # 2 index_flex
        True,   # 3 middle_flex
        True,   # 4 ring_flex
        True,   # 5 pinky_flex
        False,  # 6 index_abd
        False,  # 7 ring_abd
        False,  # 8 pinky_abd
        True,   # 9 thumb_yaw
    ],
    dtype=bool,
)


def _as_10d(values) -> np.ndarray:
    """
    Convert input to float64 vector with shape (10,).
    """

    q = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if q.shape != (10,):
        raise ValueError(
            "L20Lite command must contain exactly "
            f"10 values, got shape={q.shape}"
        )

    return q


# ============================================================
# Real SDK -> MuJoCo normalized 0~100
# ============================================================

def real_sdk_to_mujoco(values):
    """
    Convert LinkerBot real-hand SDK angles (0~100)
    into the normalized 0~100 convention currently
    used by l20lite_sim.py.

    This handles:
        - semantic channel correspondence
        - experimentally verified polarity

    It does NOT yet perform additional endpoint/offset
    calibration beyond the normalized 0~100 assumption.
    """

    q_real = np.clip(
        _as_10d(values),
        0.0,
        100.0,
    )

    q_sim = q_real.copy()

    q_sim[INVERT_MASK] = (
        100.0
        - q_real[INVERT_MASK]
    )

    return q_sim


# ============================================================
# MuJoCo normalized 0~100 -> Real SDK
# ============================================================

def mujoco_to_real_sdk(values):
    """
    Inverse conversion.

    Because the current polarity transform is simply
    either identity or (100-x), the inverse transform
    has the same form.
    """

    q_sim = np.clip(
        _as_10d(values),
        0.0,
        100.0,
    )

    q_real = q_sim.copy()

    q_real[INVERT_MASK] = (
        100.0
        - q_sim[INVERT_MASK]
    )

    return q_real


# ============================================================
# Dictionary helpers
# ============================================================

def real_sdk_dict(values):
    q = _as_10d(values)

    return {
        name: float(q[i])
        for i, name in enumerate(
            REAL_SDK_ORDER
        )
    }


def mujoco_dict(values):
    q = _as_10d(values)

    return {
        name: float(q[i])
        for i, name in enumerate(
            MUJOCO_JOINT_ORDER
        )
    }


# ============================================================
# Pretty diagnostic
# ============================================================

def print_mapping(real_values):
    q_real = _as_10d(real_values)

    q_sim = real_sdk_to_mujoco(
        q_real
    )

    print(
        "=============================================================="
    )

    print(
        " L20Lite RIGHT real -> MuJoCo mapping"
    )

    print(
        "=============================================================="
    )

    for i in range(10):

        direction = (
            "INVERT"
            if INVERT_MASK[i]
            else "SAME"
        )

        print(
            f"[{i}] "
            f"{REAL_SDK_ORDER[i]:<14s} "
            f"{q_real[i]:7.2f} "
            f"-> "
            f"{MUJOCO_JOINT_ORDER[i]:<22s} "
            f"{q_sim[i]:7.2f} "
            f"({direction})"
        )


# ============================================================

if __name__ == "__main__":

    # Example based on the measured real right-hand posture.
    example = np.array(
        [
            99.22,
            48.63,
            99.61,
            99.61,
            99.61,
            99.61,
            4.31,
            6.67,
            0.00,
            43.92,
        ],
        dtype=np.float64,
    )

    print_mapping(
        example
    )

    q_sim = real_sdk_to_mujoco(
        example
    )

    q_back = mujoco_to_real_sdk(
        q_sim
    )

    print()
    print(
        "Round-trip max error:",
        float(
            np.max(
                np.abs(
                    q_back
                    - example
                )
            )
        ),
    )
