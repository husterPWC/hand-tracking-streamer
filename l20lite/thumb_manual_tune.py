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
    / "dual_l20lite_dynamic.xml"
)


# ============================================================
# Approximate physical tip point inside distal mesh frame
# ============================================================

THUMB_TIP_LOCAL = np.array(
    [0.0, 0.0, 0.023],
    dtype=np.float64,
)

INDEX_TIP_LOCAL = np.array(
    [0.0, 0.0, 0.023],
    dtype=np.float64,
)


def geom_id(
    model,
    name,
):
    """
    Resolve a hand geom robustly.

    dynamic.xml does not assign explicit names such as:
        right_thumb_distal_geom

    but its mesh asset is still named:
        right_thumb_distal

    Therefore:
      1. try explicit geom name first
      2. otherwise resolve the geom through its mesh name
    """

    # --------------------------------------------------------
    # 1. Try explicit geom name
    # --------------------------------------------------------

    gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )

    if gid >= 0:
        return gid

    # --------------------------------------------------------
    # 2. Convert:
    #
    # right_thumb_distal_geom
    # ->
    # right_thumb_distal
    # --------------------------------------------------------

    mesh_name = name

    if mesh_name.endswith(
        "_geom"
    ):
        mesh_name = mesh_name[
            :-5
        ]

    mesh_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_MESH,
        mesh_name,
    )

    if mesh_id < 0:

        raise KeyError(
            f"Missing geom '{name}' "
            f"and mesh '{mesh_name}'"
        )

    # --------------------------------------------------------
    # Find which mesh geom uses this mesh asset.
    # geom_dataid stores the mesh id for mesh-type geoms.
    # --------------------------------------------------------

    for candidate_gid in range(
        model.ngeom
    ):

        if (
            model.geom_type[
                candidate_gid
            ]
            == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom_dataid[
                candidate_gid
            ]
            == mesh_id
        ):

            return candidate_gid

    raise KeyError(
        f"Mesh '{mesh_name}' exists, "
        "but no geom uses it."
    )


def geom_local_point_to_world(
    data,
    gid,
    local_point,
):

    p = np.asarray(
        data.geom_xpos[gid],
        dtype=np.float64,
    )

    R = np.asarray(
        data.geom_xmat[gid],
        dtype=np.float64,
    ).reshape(3, 3)

    return (
        p
        + R
        @ np.asarray(
            local_point,
            dtype=np.float64,
        )
    )


def main():

    sim = L20LiteMuJoCo(
        MODEL_PATH
    )

    model = sim.model
    data = sim.data

    thumb_gid = geom_id(
        model,
        "right_thumb_distal_geom",
    )

    index_gid = geom_id(
        model,
        "right_index_distal_geom",
    )

    # --------------------------------------------------------
    # Natural relaxed hand.
    #
    # 0 thumb pitch
    # 1 thumb yaw
    # 2 index flex
    # ...
    # 9 thumb roll
    # --------------------------------------------------------

    command = np.array(
        [
            0,      # thumb pitch
            0,      # thumb yaw

            15,     # index slight flex

            10,     # middle
            12,     # ring
            15,     # pinky

            0,
            0,
            0,

            0,      # thumb roll
        ],
        dtype=np.float64,
    )

    # Shared mutable state for keyboard callback.
    state = {
        "command": command,
        "changed": True,
    }

    STEP = 5.0

    def print_state():

        q = state[
            "command"
        ]

        mujoco.mj_forward(
            model,
            data,
        )

        thumb_tip = (
            geom_local_point_to_world(
                data,
                thumb_gid,
                THUMB_TIP_LOCAL,
            )
        )

        index_tip = (
            geom_local_point_to_world(
                data,
                index_gid,
                INDEX_TIP_LOCAL,
            )
        )

        gap = np.linalg.norm(
            thumb_tip
            - index_tip
        )

        print(
            "\n----------------------------------------"
        )

        print(
            "THUMB"
        )

        print(
            f"  pitch [0] = {q[0]:5.1f}"
        )

        print(
            f"  yaw   [1] = {q[1]:5.1f}"
        )

        print(
            f"  roll  [9] = {q[9]:5.1f}"
        )

        print(
            f"INDEX"
        )

        print(
            f"  flex  [2] = {q[2]:5.1f}"
        )

        print(
            "thumb tip:",
            np.round(
                thumb_tip,
                4,
            ).tolist()
        )

        print(
            "index tip:",
            np.round(
                index_tip,
                4,
            ).tolist()
        )

        print(
            f"tip gap: {gap*1000:.1f} mm"
        )

        print(
            "command:",
            np.round(
                q,
                1,
            ).tolist()
        )

        print(
            "----------------------------------------"
        )

    def change(index, delta):

        state[
            "command"
        ][index] = np.clip(
            state[
                "command"
            ][index]
            + delta,
            0.0,
            100.0,
        )

        state[
            "changed"
        ] = True

    def key_callback(keycode):

        try:
            key = chr(
                keycode
            ).lower()

        except Exception:
            return

        # --------------------------------------------
        # Thumb pitch
        # Q = increase
        # A = decrease
        # --------------------------------------------

        if key == "q":
            change(
                0,
                +STEP,
            )

        elif key == "a":
            change(
                0,
                -STEP,
            )

        # --------------------------------------------
        # Thumb yaw
        # W = increase
        # S = decrease
        # --------------------------------------------

        elif key == "w":
            change(
                1,
                +STEP,
            )

        elif key == "s":
            change(
                1,
                -STEP,
            )

        # --------------------------------------------
        # Thumb roll
        # E = increase
        # D = decrease
        # --------------------------------------------

        elif key == "e":
            change(
                9,
                +STEP,
            )

        elif key == "d":
            change(
                9,
                -STEP,
            )

        # --------------------------------------------
        # Index flex
        # Z/X
        # --------------------------------------------

        elif key == "x":
            change(
                2,
                +STEP,
            )

        elif key == "z":
            change(
                2,
                -STEP,
            )

        # --------------------------------------------
        # Reset thumb
        # --------------------------------------------

        elif key == "r":

            state[
                "command"
            ][0] = 0

            state[
                "command"
            ][1] = 0

            state[
                "command"
            ][9] = 0

            state[
                "changed"
            ] = True

        # --------------------------------------------
        # Print current pose
        # --------------------------------------------

        elif key == "p":

            print_state()

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    sim.left.angle.set_angles(
        [0] * 10
    )

    sim.right.angle.set_angles(
        command
    )

    print(
        "========================================"
    )

    print(
        " L20 Lite Thumb Manual Calibration"
    )

    print(
        "========================================"
    )

    print(
        """
Click the MuJoCo window first.

Keyboard:

  Q / A : thumb pitch + / -
  W / S : thumb yaw   + / -
  E / D : thumb roll  + / -

  X / Z : index flex  + / -

  R     : reset thumb to 0
  P     : print current values

Each key changes 5%.
"""
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:

        viewer.cam.lookat[:] = np.array(
            [
                0.02,
                -0.18,
                0.13,
            ],
            dtype=np.float64,
        )

        viewer.cam.distance = 0.27
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -5

        viewer.sync()

        counter = 0

        while viewer.is_running():

            sim.right.angle.set_angles(
                state[
                    "command"
                ]
            )

            sim.step()

            counter += 1

            if (
                state[
                    "changed"
                ]
            ):

                print_state()

                state[
                    "changed"
                ] = False

            if counter % 10 == 0:

                viewer.sync()

            time.sleep(
                sim.timestep
            )


if __name__ == "__main__":
    main()
