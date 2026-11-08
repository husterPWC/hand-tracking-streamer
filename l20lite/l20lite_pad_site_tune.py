#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

CALIB_PATH = (
    ROOT
    / "calibration"
    / "l20lite_pad_sites.json"
)


FINGERS = (
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
)


DEFAULT_PAD = np.array(
    [
        0.0,
        0.0,
        0.013,
    ],
    dtype=np.float64,
)


# One key press = 0.5 mm.
STEP = 0.0005


# ============================================================
# Helpers
# ============================================================

def quat_to_mat(
    quat,
):

    quat = np.asarray(
        quat,
        dtype=np.float64,
    )

    mat = np.zeros(
        9,
        dtype=np.float64,
    )

    mujoco.mju_quat2Mat(
        mat,
        quat,
    )

    return mat.reshape(
        3,
        3,
    )


def resolve_geom_id_from_mesh(
    model,
    mesh_name,
):

    mesh_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_MESH,
        mesh_name,
    )

    if mesh_id < 0:

        raise RuntimeError(
            f"Mesh missing: "
            f"{mesh_name}"
        )

    for gid in range(
        model.ngeom
    ):

        if (
            model.geom_type[
                gid
            ]
            == mujoco.mjtGeom.mjGEOM_MESH
            and
            model.geom_dataid[
                gid
            ]
            == mesh_id
        ):

            return gid

    raise RuntimeError(
        f"No geom uses mesh: "
        f"{mesh_name}"
    )


def default_calibration():

    result = {}

    for side in (
        "left",
        "right",
    ):

        result[side] = {}

        for finger in FINGERS:

            result[
                side
            ][finger] = (
                DEFAULT_PAD.copy()
            )

    return result


def load_calibration():

    result = (
        default_calibration()
    )

    if not CALIB_PATH.exists():

        return result

    try:

        with open(
            CALIB_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            loaded = json.load(
                f
            )

        for side in (
            "left",
            "right",
        ):

            for finger in FINGERS:

                if (
                    side in loaded
                    and finger
                    in loaded[
                        side
                    ]
                ):

                    value = np.asarray(
                        loaded[
                            side
                        ][
                            finger
                        ],
                        dtype=np.float64,
                    )

                    if value.shape == (3,):

                        result[
                            side
                        ][
                            finger
                        ] = value

    except Exception as exc:

        print(
            f"[WARN] Could not load "
            f"calibration: {exc}"
        )

    return result


def save_calibration(
    calibration,
):

    CALIB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable = {}

    for side in (
        "left",
        "right",
    ):

        serializable[
            side
        ] = {}

        for finger in FINGERS:

            serializable[
                side
            ][
                finger
            ] = (
                np.asarray(
                    calibration[
                        side
                    ][
                        finger
                    ],
                    dtype=np.float64,
                )
                .tolist()
            )

    with open(
        CALIB_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            serializable,
            f,
            indent=2,
        )

    print(
        "\n[SAVED]"
    )

    print(
        f"  {CALIB_PATH}"
    )


# ============================================================
# Site update
# ============================================================

def update_one_pad_site(
    sim,
    calibration,
    side,
    finger,
):

    model = sim.model

    mesh_name = (
        f"{side}_{finger}_distal"
    )

    site_name = (
        f"{side}_{finger}_pad_site"
    )

    gid = resolve_geom_id_from_mesh(
        model,
        mesh_name,
    )

    sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if sid < 0:

        raise RuntimeError(
            f"Missing site: "
            f"{site_name}"
        )

    local_point = np.asarray(
        calibration[
            side
        ][
            finger
        ],
        dtype=np.float64,
    )

    geom_pos = np.asarray(
        model.geom_pos[
            gid
        ],
        dtype=np.float64,
    )

    geom_rot = quat_to_mat(
        model.geom_quat[
            gid
        ]
    )

    # site_pos is relative to the same parent body.
    model.site_pos[
        sid
    ] = (
        geom_pos
        + geom_rot
        @ local_point
    )


def update_all_pad_sites(
    sim,
    calibration,
):

    for side in (
        "left",
        "right",
    ):

        for finger in FINGERS:

            update_one_pad_site(
                sim,
                calibration,
                side,
                finger,
            )

    mujoco.mj_forward(
        sim.model,
        sim.data,
    )


# ============================================================
# Print state
# ============================================================

def print_current(
    sim,
    calibration,
    side,
    finger,
):

    local = np.asarray(
        calibration[
            side
        ][
            finger
        ],
        dtype=np.float64,
    )

    site_name = (
        f"{side}_{finger}_pad_site"
    )

    sid = mujoco.mj_name2id(
        sim.model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    world = np.asarray(
        sim.data.site_xpos[
            sid
        ],
        dtype=np.float64,
    )

    print(
        "\n========================================"
    )

    print(
        f"Selected: "
        f"{side.upper()} / "
        f"{finger.upper()}"
    )

    print(
        "----------------------------------------"
    )

    print(
        "geom-local pad position:"
    )

    print(
        f"  x = "
        f"{local[0]*1000:+7.2f} mm"
    )

    print(
        f"  y = "
        f"{local[1]*1000:+7.2f} mm"
    )

    print(
        f"  z = "
        f"{local[2]*1000:+7.2f} mm"
    )

    print(
        "world position:"
    )

    print(
        " ",
        np.round(
            world,
            4,
        ).tolist()
    )

    print(
        "========================================"
    )


def print_all(
    calibration,
):

    print(
        "\n========================================"
    )

    print(
        " CURRENT PAD CALIBRATION"
    )

    print(
        "========================================"
    )

    for side in (
        "left",
        "right",
    ):

        print(
            f"\n[{side.upper()}]"
        )

        for finger in FINGERS:

            p = np.asarray(
                calibration[
                    side
                ][
                    finger
                ],
                dtype=np.float64,
            )

            print(
                f"{finger:<7s}: "
                f"["
                f"{p[0]:+.5f}, "
                f"{p[1]:+.5f}, "
                f"{p[2]:+.5f}"
                f"]"
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hand",
        choices=(
            "left",
            "right",
        ),
        default="right",
    )

    args = parser.parse_args()

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"{MODEL_PATH}\n"
            "Run build_l20lite_tactile.py first."
        )

    sim = L20LiteMuJoCo(
        MODEL_PATH
    )

    calibration = (
        load_calibration()
    )

    state = {
        "side":
            args.hand,

        "finger_index":
            0,

        "changed":
            True,
    }


    def current_finger():

        return FINGERS[
            state[
                "finger_index"
            ]
        ]


    def refresh():

        update_all_pad_sites(
            sim,
            calibration,
        )

        # ====================================================
        # Visual highlighting
        #
        # All pad sites:
        #     GREEN
        #
        # Currently selected pad:
        #     YELLOW + larger
        # ====================================================

        for side_name in (
            "left",
            "right",
        ):

            for finger_name in FINGERS:

                site_name = (
                    f"{side_name}_"
                    f"{finger_name}_"
                    f"pad_site"
                )

                sid = mujoco.mj_name2id(
                    sim.model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    site_name,
                )

                if sid < 0:
                    continue

                # Normal pad marker = green
                sim.model.site_rgba[
                    sid
                ] = np.array(
                    [
                        0.1,
                        1.0,
                        0.1,
                        1.0,
                    ],
                    dtype=np.float32,
                )

                sim.model.site_size[
                    sid,
                    0
                ] = 0.004

        # Selected pad = yellow
        selected_name = (
            f"{state['side']}_"
            f"{current_finger()}_"
            f"pad_site"
        )

        selected_sid = mujoco.mj_name2id(
            sim.model,
            mujoco.mjtObj.mjOBJ_SITE,
            selected_name,
        )

        if selected_sid >= 0:

            sim.model.site_rgba[
                selected_sid
            ] = np.array(
                [
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            )

            sim.model.site_size[
                selected_sid,
                0
            ] = 0.005

        mujoco.mj_forward(
            sim.model,
            sim.data,
        )

        print_current(
            sim,
            calibration,
            state[
                "side"
            ],
            current_finger(),
        )


    def move_selected(
        axis,
        delta,
    ):

        side = state[
            "side"
        ]

        finger = (
            current_finger()
        )

        calibration[
            side
        ][
            finger
        ][
            axis
        ] += delta

        state[
            "changed"
        ] = True


    def key_callback(
        keycode,
    ):

        try:

            key = chr(
                keycode
            ).lower()

        except Exception:

            return

        # ================================================
        # Select finger
        # ================================================

        if key in (
            "1",
            "2",
            "3",
            "4",
            "5",
        ):

            state[
                "finger_index"
            ] = (
                int(key)
                - 1
            )

            state[
                "changed"
            ] = True

        # ================================================
        # Local X
        # ================================================

        elif key == "q":

            move_selected(
                0,
                +STEP,
            )

        elif key == "a":

            move_selected(
                0,
                -STEP,
            )

        # ================================================
        # Local Y
        # ================================================

        elif key == "w":

            move_selected(
                1,
                +STEP,
            )

        elif key == "s":

            move_selected(
                1,
                -STEP,
            )

        # ================================================
        # Local Z
        # ================================================

        elif key == "e":

            move_selected(
                2,
                +STEP,
            )

        elif key == "d":

            move_selected(
                2,
                -STEP,
            )

        # ================================================
        # Toggle left/right hand
        # ================================================

        elif key == "h":

            state[
                "side"
            ] = (
                "left"
                if state[
                    "side"
                ]
                == "right"
                else "right"
            )

            state[
                "changed"
            ] = True

        # ================================================
        # Reset selected finger
        # ================================================

        elif key == "r":

            calibration[
                state[
                    "side"
                ]
            ][
                current_finger()
            ] = (
                DEFAULT_PAD.copy()
            )

            state[
                "changed"
            ] = True

        # ================================================
        # Print all
        # ================================================

        elif key == "p":

            print_all(
                calibration
            )

        # ================================================
        # Save calibration
        # ================================================

        elif key == "k":

            save_calibration(
                calibration
            )


    # ----------------------------------------------------
    # Keep both hands in a relaxed/open configuration.
    # ----------------------------------------------------

    left_cmd = np.array(
        [
            0, 0,
            5, 8, 10, 12,
            0, 0, 0,
            0,
        ],
        dtype=np.float64,
    )

    right_cmd = left_cmd.copy()

    sim.left.angle.set_angles(
        left_cmd
    )

    sim.right.angle.set_angles(
        right_cmd
    )

    update_all_pad_sites(
        sim,
        calibration,
    )

    print(
        "========================================"
    )

    print(
        " L20 Lite Five-Finger Pad-Site Tuner"
    )

    print(
        "========================================"
    )

    print(
        """
RED   = fingertip reference
GREEN = tactile pad center

Click the MuJoCo viewer first.

Finger selection:
  1 = thumb
  2 = index
  3 = middle
  4 = ring
  5 = pinky

Move selected GREEN pad in distal-geom LOCAL frame:
  Q / A : local X  + / -
  W / S : local Y  + / -
  E / D : local Z  + / -

Each press = 0.5 mm

Other:
  H = switch LEFT / RIGHT hand
  R = reset selected pad
  P = print all pad coordinates
  K = SAVE calibration JSON

Goal:
  Move every GREEN marker to the center
  of the physical tactile-film / finger-pad area.
"""
    )

    refresh()

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=key_callback,
    ) as viewer:

        viewer.opt.sitegroup[:] = 1

        if args.hand == "right":

            viewer.cam.lookat[:] = (
                np.array(
                    [
                        0.02,
                        -0.18,
                        0.14,
                    ]
                )
            )

        else:

            viewer.cam.lookat[:] = (
                np.array(
                    [
                        0.02,
                        0.18,
                        0.14,
                    ]
                )
            )

        viewer.cam.distance = 0.27
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -8

        viewer.sync()

        counter = 0

        while viewer.is_running():

            sim.left.angle.set_angles(
                left_cmd
            )

            sim.right.angle.set_angles(
                right_cmd
            )

            sim.step()

            counter += 1

            if state[
                "changed"
            ]:

                refresh()

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
