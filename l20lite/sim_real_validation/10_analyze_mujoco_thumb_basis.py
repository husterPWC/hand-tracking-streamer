#!/usr/bin/env python3

from pathlib import Path
import sys

import mujoco
import numpy as np


MODEL_PATH = PROJECT_ROOT / "generated" / "dual_l20lite_dynamic.xml"

SIDE = "right"

ACTIVE = [
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "thumb_cmc_roll",
]

# Neutral-ish posture based on what we have already established:
#
# real thumb_flex ~100 -> sim cmc_yaw ~0
# real thumb_abd  ~50  -> sim cmc_pitch ~50%
# thumb roll still unknown -> start from 0
REF_PERCENT = {
    "thumb_cmc_pitch": 50.0,
    "thumb_cmc_yaw":    0.0,
    "thumb_cmc_roll":   0.0,
}

EPS = 1e-3


def joint_id(model, short_name):
    name = f"{SIDE}_{short_name}"

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(f"Joint not found: {name}")

    return jid


def set_joint(model, data, short_name, q):
    jid = joint_id(model, short_name)
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr] = q


def get_range(model, short_name):
    jid = joint_id(model, short_name)

    lo, hi = model.jnt_range[jid]

    return float(lo), float(hi)


def pct_to_rad(model, short_name, pct):
    lo, hi = get_range(model, short_name)

    return lo + pct / 100.0 * (hi - lo)


def set_thumb(model, data, pitch, yaw, roll):
    # Active joints
    set_joint(
        model,
        data,
        "thumb_cmc_pitch",
        pitch,
    )

    set_joint(
        model,
        data,
        "thumb_cmc_yaw",
        yaw,
    )

    set_joint(
        model,
        data,
        "thumb_cmc_roll",
        roll,
    )

    # URDF mimic constraints.
    set_joint(
        model,
        data,
        "thumb_mcp",
        1.38 * pitch,
    )

    set_joint(
        model,
        data,
        "thumb_ip",
        1.49 * pitch,
    )

    mujoco.mj_forward(
        model,
        data,
    )


def body_pose(model, data):
    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"{SIDE}_thumb_distal",
    )

    if bid < 0:
        raise RuntimeError(
            "right_thumb_distal body not found"
        )

    p = data.xpos[bid].copy()

    R = (
        data.xmat[bid]
        .reshape(3, 3)
        .copy()
    )

    return p, R


def rotation_vector_small(R0, R1):
    """
    Small-angle world-frame rotation vector.
    """

    dR = R1 @ R0.T

    w = np.array(
        [
            dR[2, 1] - dR[1, 2],
            dR[0, 2] - dR[2, 0],
            dR[1, 0] - dR[0, 1],
        ]
    ) * 0.5

    return w


def main():
    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    # --------------------------------------------------------
    # Reference thumb configuration
    # --------------------------------------------------------

    q_ref = {}

    for name in ACTIVE:
        q_ref[name] = pct_to_rad(
            model,
            name,
            REF_PERCENT[name],
        )

    set_thumb(
        model,
        data,
        q_ref["thumb_cmc_pitch"],
        q_ref["thumb_cmc_yaw"],
        q_ref["thumb_cmc_roll"],
    )

    p0, R0 = body_pose(
        model,
        data,
    )

    # Local Z of distal thumb.
    # This is our current approximation
    # of thumb longitudinal direction.
    thumb_axis = R0[:, 2].copy()
    thumb_axis /= np.linalg.norm(
        thumb_axis
    )

    print("=" * 72)
    print("L20Lite RIGHT thumb 3-DoF kinematic basis")
    print("=" * 72)

    print("\nReference:")
    for name in ACTIVE:
        print(
            f"  {name:<20s}: "
            f"{q_ref[name]:.6f} rad "
            f"({REF_PERCENT[name]:.1f}%)"
        )

    print(
        "\nThumb distal longitudinal axis "
        "(world approximation):"
    )
    print(
        np.round(
            thumb_axis,
            6,
        )
    )

    # --------------------------------------------------------
    # Finite-difference Jacobian
    # --------------------------------------------------------

    Jv = np.zeros((3, 3))
    Jw = np.zeros((3, 3))

    for col, name in enumerate(ACTIVE):

        q = dict(q_ref)
        q[name] += EPS

        set_thumb(
            model,
            data,
            q["thumb_cmc_pitch"],
            q["thumb_cmc_yaw"],
            q["thumb_cmc_roll"],
        )

        p1, R1 = body_pose(
            model,
            data,
        )

        dp = (p1 - p0) / EPS

        dw = (
            rotation_vector_small(
                R0,
                R1,
            )
            / EPS
        )

        Jv[:, col] = dp
        Jw[:, col] = dw

    print("\n--------------------------------------------------------")
    print("Linear velocity basis Jv")
    print("columns = [pitch, yaw, roll]")
    print("--------------------------------------------------------")
    print(
        np.array2string(
            Jv,
            precision=6,
            suppress_small=True,
        )
    )

    print("\n--------------------------------------------------------")
    print("Angular velocity basis Jw")
    print("columns = [pitch, yaw, roll]")
    print("--------------------------------------------------------")
    print(
        np.array2string(
            Jw,
            precision=6,
            suppress_small=True,
        )
    )

    # --------------------------------------------------------
    # How much each joint resembles axial thumb rotation
    # --------------------------------------------------------

    print("\n--------------------------------------------------------")
    print("Axial rotation contribution")
    print("--------------------------------------------------------")

    for i, name in enumerate(ACTIVE):

        axial = float(
            np.dot(
                Jw[:, i],
                thumb_axis,
            )
        )

        transl = float(
            np.linalg.norm(
                Jv[:, i]
            )
        )

        print(
            f"{name:<20s} "
            f"axial={axial:+.6f} "
            f"translation={transl:.6f} m/rad"
        )

    # --------------------------------------------------------
    # Find joint combination:
    #
    # desired:
    #   angular velocity ~= thumb longitudinal axis
    #   translation ~= 0
    #
    # Translation is weighted because meters and radians
    # have different numerical scales.
    # --------------------------------------------------------

    TRANSLATION_WEIGHT = 20.0

    A = np.vstack(
        [
            Jw,
            TRANSLATION_WEIGHT * Jv,
        ]
    )

    b = np.concatenate(
        [
            thumb_axis,
            np.zeros(3),
        ]
    )

    dq, *_ = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    # Normalize for easy interpretation.
    scale = np.max(np.abs(dq))

    if scale > 1e-12:
        dq_norm = dq / scale
    else:
        dq_norm = dq

    print("\n========================================================")
    print("Candidate MuJoCo thumb axial-rotation combination")
    print("========================================================")

    print(
        "Raw dq [pitch, yaw, roll] =",
        np.round(dq, 6),
    )

    print(
        "Normalized combination     =",
        np.round(dq_norm, 4),
    )

    predicted_v = Jv @ dq
    predicted_w = Jw @ dq

    axial_component = np.dot(
        predicted_w,
        thumb_axis,
    )

    print(
        "\nPredicted translation norm = "
        f"{np.linalg.norm(predicted_v):.6f} m"
    )

    print(
        "Predicted angular velocity =",
        np.round(
            predicted_w,
            6,
        ),
    )

    print(
        "Predicted axial component  = "
        f"{axial_component:.6f}"
    )

    print("\nInterpretation:")
    print(
        "  Real thumb_yaw should NOT be hard-mapped "
        "to thumb_cmc_roll."
    )
    print(
        "  The normalized combination above is the "
        "candidate MuJoCo basis for thumb axial rotation."
    )


if __name__ == "__main__":
    main()
