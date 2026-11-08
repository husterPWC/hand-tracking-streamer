#!/usr/bin/env python3
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent

INPUT_PATH = (
    ROOT
    / "generated"
    / "dual_l20lite_dynamic.xml"
)

OUTPUT_PATH = (
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


# ============================================================
# Geom-local positions
# ============================================================

# Already visually verified:
# red fingertip reference point.
TIP_LOCAL_IN_GEOM = np.array(
    [
        0.0,
        0.0,
        0.013,
    ],
    dtype=np.float64,
)


# Initial green pad marker.
#
# Start slightly behind the fingertip apex.
# We will manually move XYZ to the physical tactile-pad center.
PAD_DEFAULT_LOCAL_IN_GEOM = np.array(
    [
        0.0,
        0.0,
        0.013,
    ],
    dtype=np.float64,
)


# ============================================================
# Quaternion -> matrix
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


# ============================================================
# XML helpers
# ============================================================

def make_parent_map(
    root,
):

    return {
        child: parent
        for parent in root.iter()
        for child in parent
    }


def find_geom_element_by_mesh(
    root,
    mesh_name,
):

    matches = []

    for geom in root.iter(
        "geom"
    ):

        if (
            geom.attrib.get(
                "mesh",
                ""
            )
            == mesh_name
        ):

            matches.append(
                geom
            )

    if len(matches) != 1:

        raise RuntimeError(
            f"Expected exactly one XML geom "
            f"using mesh '{mesh_name}', "
            f"found {len(matches)}"
        )

    return matches[0]


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
            f"Mesh not found: "
            f"{mesh_name}"
        )

    matches = []

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

            matches.append(
                gid
            )

    if len(matches) != 1:

        raise RuntimeError(
            f"Expected one compiled geom "
            f"for mesh '{mesh_name}', "
            f"found {len(matches)}"
        )

    return matches[0]


# ============================================================
# Calibration
# ============================================================

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
                PAD_DEFAULT_LOCAL_IN_GEOM
                .copy()
                .tolist()
            )

    return result


def load_calibration():

    calibration = (
        default_calibration()
    )

    if not CALIB_PATH.exists():

        print(
            "[CALIB] No saved pad calibration."
        )

        print(
            "[CALIB] Using default pad positions."
        )

        return calibration

    print(
        f"[CALIB] Loading:\n"
        f"  {CALIB_PATH}"
    )

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

            try:

                value = np.asarray(
                    loaded[
                        side
                    ][
                        finger
                    ],
                    dtype=np.float64,
                )

                if value.shape == (3,):

                    calibration[
                        side
                    ][
                        finger
                    ] = (
                        value.tolist()
                    )

            except Exception:

                pass

    return calibration


# ============================================================
# Add site
# ============================================================

def add_site(
    root,
    parent_map,
    model,
    side,
    finger,
    suffix,
    local_point_geom,
    rgba,
    size,
):

    mesh_name = (
        f"{side}_{finger}_distal"
    )

    site_name = (
        f"{side}_{finger}_{suffix}"
    )

    gid = resolve_geom_id_from_mesh(
        model,
        mesh_name,
    )

    # Geom transform relative to parent body.
    geom_pos = np.asarray(
        model.geom_pos[
            gid
        ],
        dtype=np.float64,
    )

    geom_quat = np.asarray(
        model.geom_quat[
            gid
        ],
        dtype=np.float64,
    )

    geom_rot = quat_to_mat(
        geom_quat
    )

    # Convert:
    #
    # geom-local point
    #     ->
    # body-local point
    #
    local_point_geom = np.asarray(
        local_point_geom,
        dtype=np.float64,
    )

    site_body_pos = (
        geom_pos
        + geom_rot
        @ local_point_geom
    )

    geom_element = (
        find_geom_element_by_mesh(
            root,
            mesh_name,
        )
    )

    parent = parent_map.get(
        geom_element
    )

    while (
        parent is not None
        and parent.tag != "body"
    ):

        parent = parent_map.get(
            parent
        )

    if parent is None:

        raise RuntimeError(
            f"Cannot find parent body "
            f"for {mesh_name}"
        )

    # Remove duplicate if present.
    for old_site in list(
        parent.findall(
            "site"
        )
    ):

        if (
            old_site.attrib.get(
                "name"
            )
            == site_name
        ):

            parent.remove(
                old_site
            )

    ET.SubElement(
        parent,
        "site",
        {
            "name":
                site_name,

            "type":
                "sphere",

            "size":
                f"{size:.6f}",

            "pos":
                (
                    f"{site_body_pos[0]:.10g} "
                    f"{site_body_pos[1]:.10g} "
                    f"{site_body_pos[2]:.10g}"
                ),

            # Keep the site frame aligned with the distal geom.
            "quat":
                (
                    f"{geom_quat[0]:.10g} "
                    f"{geom_quat[1]:.10g} "
                    f"{geom_quat[2]:.10g} "
                    f"{geom_quat[3]:.10g}"
                ),

            "rgba":
                rgba,

            "group":
                "3",
        }
    )

    return (
        site_name,
        site_body_pos,
    )


# ============================================================
# Validation
# ============================================================

def validate():

    model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                OUTPUT_PATH
            )
        )
    )

    data = mujoco.MjData(
        model
    )

    mujoco.mj_forward(
        model,
        data,
    )

    print(
        "\n========================================"
    )

    print(
        "[TACTILE MODEL]"
    )

    print(
        f"nq      = {model.nq}"
    )

    print(
        f"nv      = {model.nv}"
    )

    print(
        f"nu      = {model.nu}"
    )

    print(
        f"neq     = {model.neq}"
    )

    print(
        f"ngeom   = {model.ngeom}"
    )

    print(
        f"nsite   = {model.nsite}"
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

            tip_name = (
                f"{side}_{finger}_tip_site"
            )

            pad_name = (
                f"{side}_{finger}_pad_site"
            )

            tip_sid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                tip_name,
            )

            pad_sid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                pad_name,
            )

            if tip_sid < 0:

                raise RuntimeError(
                    f"Missing {tip_name}"
                )

            if pad_sid < 0:

                raise RuntimeError(
                    f"Missing {pad_name}"
                )

            print(
                f"  {finger:<7s}: "
                f"tip={tip_sid:2d}, "
                f"pad={pad_sid:2d}"
            )

    if model.nsite != 20:

        raise RuntimeError(
            f"Expected exactly 20 sites, "
            f"got {model.nsite}"
        )

    print(
        "\n[SUCCESS]"
    )

    print(
        "10 fingertip sites + "
        "10 tactile pad sites are ready."
    )

    print(
        f"Saved to:\n"
        f"  {OUTPUT_PATH}"
    )


# ============================================================
# Main
# ============================================================

def main():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"{INPUT_PATH}\n"
            "Run build_l20lite_dynamic.py first."
        )

    print(
        "========================================"
    )

    print(
        " L20 Lite Fingertip / Tactile Builder"
    )

    print(
        "========================================"
    )

    print(
        f"Input:\n"
        f"  {INPUT_PATH}"
    )

    model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                INPUT_PATH
            )
        )
    )

    root = (
        ET.parse(
            INPUT_PATH
        ).getroot()
    )

    parent_map = (
        make_parent_map(
            root
        )
    )

    calibration = (
        load_calibration()
    )

    print(
        "\n[ADDING SITES]"
    )

    for side in (
        "left",
        "right",
    ):

        print(
            f"\n  [{side.upper()}]"
        )

        for finger in FINGERS:

            # ----------------------------------------------
            # RED = geometric fingertip reference
            # ----------------------------------------------

            tip_name, _ = add_site(
                root=root,
                parent_map=parent_map,
                model=model,
                side=side,
                finger=finger,
                suffix="tip_site",
                local_point_geom=(
                    TIP_LOCAL_IN_GEOM
                ),
                rgba="1 0.1 0.1 1",
                size=0.002,
            )

            # ----------------------------------------------
            # GREEN = tactile-pad center
            # ----------------------------------------------

            pad_local = np.asarray(
                calibration[
                    side
                ][
                    finger
                ],
                dtype=np.float64,
            )

            pad_name, _ = add_site(
                root=root,
                parent_map=parent_map,
                model=model,
                side=side,
                finger=finger,
                suffix="pad_site",
                local_point_geom=pad_local,
                rgba="0.1 1 0.1 1",
                size=0.005,
            )

            print(
                f"    {finger:<7s}"
                f" tip={tip_name}"
                f" pad={pad_name}"
                f" local="
                f"{np.round(pad_local, 4).tolist()}"
            )

    ET.indent(
        root,
        space="  ",
    )

    ET.ElementTree(
        root
    ).write(
        OUTPUT_PATH,
        encoding="utf-8",
        xml_declaration=True,
    )

    validate()


if __name__ == "__main__":
    main()
