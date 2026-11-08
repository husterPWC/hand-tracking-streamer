#!/usr/bin/env python3
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent

INPUT_PATH = (
    ROOT
    / "generated"
    / "dual_l20lite_dynamic.xml"
)

OUTPUT_PATH = (
    ROOT
    / "generated"
    / "dual_l20lite_grasp.xml"
)


# ============================================================
# Collision groups
# ============================================================

HAND_TYPE = 1
OBJECT_TYPE = 2


def configure_hand_contact(
    root: ET.Element,
):
    """
    Allow:
        hand <-> object

    Disable:
        hand <-> hand
        finger <-> finger
        left hand <-> right hand
    """

    print(
        "[1/4] Configuring hand-object collision..."
    )

    left_count = 0
    right_count = 0

    for geom in root.iter("geom"):

        mesh_name = geom.attrib.get(
            "mesh",
            "",
        )

        if mesh_name.startswith(
            "left_"
        ):
            left_count += 1

        elif mesh_name.startswith(
            "right_"
        ):
            right_count += 1

        else:
            continue

        geom.attrib["name"] = (
            f"{mesh_name}_geom"
        )

        geom.attrib["contype"] = str(
            HAND_TYPE
        )

        geom.attrib["conaffinity"] = str(
            OBJECT_TYPE
        )

        geom.attrib["condim"] = "3"

        # Initial simulation friction.
        # Later we can calibrate against the real hand.
        geom.attrib["friction"] = (
            "1.2 0.01 0.001"
        )

    print(
        f"      left geoms : {left_count}"
    )

    print(
        f"      right geoms: {right_count}"
    )

    if (
        left_count != 21
        or right_count != 21
    ):
        raise RuntimeError(
            "Expected 21 geoms per hand. "
            f"Got left={left_count}, "
            f"right={right_count}"
        )


def add_grasp_target(
    root: ET.Element,
):
    """
    Add an invisible mocap target.

    During closure:
        free cylinder --weld--> mocap target

    After bilateral contact:
        weld OFF
        gravity ON

    Then the cylinder is truly free.
    """

    print(
        "[2/4] Adding grasp target..."
    )

    worldbody = root.find(
        "worldbody"
    )

    if worldbody is None:
        raise RuntimeError(
            "No <worldbody>"
        )

    ET.SubElement(
        worldbody,
        "body",
        {
            "name":
                "grasp_target_mocap",

            "mocap":
                "true",

            "pos":
                "0 -0.18 0.18",
        }
    )


def add_free_cylinder(
    root: ET.Element,
):
    print(
        "[3/4] Adding free cylinder..."
    )

    worldbody = root.find(
        "worldbody"
    )

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name":
                "grasp_cylinder",

            # Placeholder only.
            # Runtime code determines final pose.
            "pos":
                "0 -0.18 0.18",
        }
    )

    ET.SubElement(
        body,
        "freejoint",
        {
            "name":
                "grasp_cylinder_free",
        }
    )

    ET.SubElement(
        body,
        "geom",
        {
            "name":
                "grasp_cylinder_geom",

            "type":
                "cylinder",

            # radius = 12 mm
            # half length = 35 mm
            # total length = 70 mm
            "size":
                "0.012 0.035",

            "mass":
                "0.040",

            "rgba":
                "0.90 0.45 0.12 1",

            "contype":
                str(
                    OBJECT_TYPE
                ),

            "conaffinity":
                str(
                    HAND_TYPE
                ),

            "condim":
                "3",

            "friction":
                "1.2 0.01 0.001",

            # Softer than raw mesh contact.
            "solref":
                "0.01 1",

            "solimp":
                "0.90 0.95 0.001",
        }
    )


def add_temporary_weld(
    root: ET.Element,
):
    """
    Temporarily hold the cylinder at the automatically
    computed pinch center.

    The weld is switched OFF at runtime after both
    thumb and index establish contact.
    """

    print(
        "[4/4] Adding temporary grasp weld..."
    )

    equality = root.find(
        "equality"
    )

    if equality is None:

        equality = ET.SubElement(
            root,
            "equality",
        )

    ET.SubElement(
        equality,
        "weld",
        {
            "name":
                "grasp_object_weld",

            "body1":
                "grasp_cylinder",

            "body2":
                "grasp_target_mocap",

            "solref":
                "0.005 1",

            "solimp":
                "0.95 0.99 0.001",
        }
    )


def save(
    root: ET.Element,
):

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


def validate():

    model = mujoco.MjModel.from_xml_path(
        str(
            OUTPUT_PATH
        )
    )

    print(
        "\n================================"
    )

    print(
        "[GRASP MODEL]"
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
        f"nmocap  = {model.nmocap}"
    )

    print(
        "================================"
    )

    # Original hand:
    #
    # nq=40
    # nv=40
    #
    # free object:
    #
    # +7 qpos
    # +6 qvel

    if model.nq != 47:
        raise RuntimeError(
            f"Expected nq=47, "
            f"got {model.nq}"
        )

    if model.nv != 46:
        raise RuntimeError(
            f"Expected nv=46, "
            f"got {model.nv}"
        )

    if model.nu != 20:
        raise RuntimeError(
            f"Expected nu=20, "
            f"got {model.nu}"
        )

    # 20 mimic constraints
    # + 1 temporary weld
    if model.neq != 21:
        raise RuntimeError(
            f"Expected neq=21, "
            f"got {model.neq}"
        )

    # 42 hand geoms + cylinder
    if model.ngeom != 43:
        raise RuntimeError(
            f"Expected ngeom=43, "
            f"got {model.ngeom}"
        )

    if model.nmocap != 1:
        raise RuntimeError(
            f"Expected nmocap=1, "
            f"got {model.nmocap}"
        )

    required_geoms = (
        "right_thumb_distal_geom",
        "right_index_distal_geom",
        "grasp_cylinder_geom",
    )

    print(
        "\n[KEY GEOMS]"
    )

    for name in required_geoms:

        gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        print(
            f"  {name:<32s} "
            f"id={gid}"
        )

        if gid < 0:
            raise RuntimeError(
                f"Missing geom: {name}"
            )

    weld_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        "grasp_object_weld",
    )

    print(
        f"\n  grasp_object_weld id="
        f"{weld_id}"
    )

    if weld_id < 0:

        raise RuntimeError(
            "Missing grasp_object_weld"
        )

    print(
        "\n[SUCCESS]"
    )

    print(
        OUTPUT_PATH
    )


def main():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            INPUT_PATH
        )

    print(
        "================================"
    )

    print(
        " L20 Lite Free Pinch Builder"
    )

    print(
        "================================"
    )

    root = ET.parse(
        INPUT_PATH
    ).getroot()

    configure_hand_contact(
        root
    )

    add_grasp_target(
        root
    )

    add_free_cylinder(
        root
    )

    add_temporary_weld(
        root
    )

    save(
        root
    )

    validate()


if __name__ == "__main__":
    main()
