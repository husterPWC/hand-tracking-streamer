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
    / "dual_l20lite_contact.xml"
)


# ============================================================
# Collision masks
#
# HAND   = bit 0 = 1
# OBJECT = bit 1 = 2
# TABLE  = bit 2 = 4
#
# Hand:
#   contype     = 1
#   conaffinity = 2
#
# Object:
#   contype     = 2
#   conaffinity = 1 | 4 = 5
#
# Table:
#   contype     = 4
#   conaffinity = 2
#
# Result:
#
# hand   <-> hand    : OFF
# hand   <-> object  : ON
# hand   <-> table   : OFF
# object <-> table   : ON
# ============================================================

HAND_TYPE = 1
OBJECT_TYPE = 2
TABLE_TYPE = 4


def configure_hand_contact(
    root: ET.Element,
):
    """
    Re-enable hand meshes ONLY for hand-object contact.

    Self-collision and left-right collision remain disabled
    because hand geom conaffinity does not contain HAND_TYPE.
    """

    print(
        "[1/4] Configuring hand collision masks..."
    )

    left_count = 0
    right_count = 0

    for geom in root.iter("geom"):

        mesh_name = geom.attrib.get(
            "mesh",
            ""
        )

        if mesh_name.startswith(
            "left_"
        ):

            side = "left"
            left_count += 1

        elif mesh_name.startswith(
            "right_"
        ):

            side = "right"
            right_count += 1

        else:
            continue

        # --------------------------------------------
        # Give every hand geom a deterministic name.
        # Example:
        #   right_index_distal
        # ->
        #   right_index_distal_geom
        # --------------------------------------------

        geom.attrib["name"] = (
            f"{mesh_name}_geom"
        )

        # --------------------------------------------
        # Collision filtering
        # --------------------------------------------

        geom.attrib["contype"] = str(
            HAND_TYPE
        )

        geom.attrib["conaffinity"] = str(
            OBJECT_TYPE
        )

        geom.attrib["condim"] = "3"

        # Initial friction setting.
        # Not yet claimed to match the physical hand.
        geom.attrib["friction"] = (
            "1.0 0.01 0.001"
        )

    print(
        f"      left hand geoms : {left_count}"
    )

    print(
        f"      right hand geoms: {right_count}"
    )

    print(
        f"      total            : "
        f"{left_count + right_count}"
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


def remove_old_test_scene(
    root: ET.Element,
):
    """
    Allows the script to be run repeatedly.
    """

    worldbody = root.find(
        "worldbody"
    )

    if worldbody is None:
        raise RuntimeError(
            "MJCF has no <worldbody>"
        )

    for name in (
        "contact_test_table",
        "contact_cylinder_mocap",
    ):

        for body in list(
            worldbody.findall("body")
        ):

            if body.attrib.get(
                "name"
            ) == name:

                worldbody.remove(
                    body
                )

    for geom in list(
        worldbody.findall("geom")
    ):

        if geom.attrib.get(
            "name"
        ) == "contact_test_table_geom":

            worldbody.remove(
                geom
            )


def add_table(
    root: ET.Element,
):
    """
    Add a table mainly as scene context.

    Hand-table contact is filtered out for this test.
    Cylinder-table contact is allowed.
    """

    print(
        "[2/4] Adding table..."
    )

    worldbody = root.find(
        "worldbody"
    )

    ET.SubElement(
        worldbody,
        "geom",
        {
            "name":
                "contact_test_table_geom",

            "type":
                "box",

            # MuJoCo box size = half extents
            "size":
                "0.35 0.35 0.01",

            "pos":
                "0 0 -0.10",

            "rgba":
                "0.45 0.45 0.45 1",

            "contype":
                str(TABLE_TYPE),

            "conaffinity":
                str(OBJECT_TYPE),

            "friction":
                "1.0 0.01 0.001",
        }
    )

    print(
        "      table z = -0.10 m"
    )


def add_mocap_cylinder(
    root: ET.Element,
):
    """
    Add a kinematic/mocap cylinder.

    It has no joint and therefore does not increase nq/nv.
    Its pose can be changed through:
        data.mocap_pos
        data.mocap_quat
    """

    print(
        "[3/4] Adding mocap cylinder..."
    )

    worldbody = root.find(
        "worldbody"
    )

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name":
                "contact_cylinder_mocap",

            "mocap":
                "true",

            # Initial placeholder.
            # Demo will automatically reposition it.
            "pos":
                "0 -0.18 0.15",
        }
    )

    ET.SubElement(
        body,
        "geom",
        {
            "name":
                "contact_cylinder_geom",

            "type":
                "cylinder",

            # radius = 15 mm
            # half-length = 30 mm
            "size":
                "0.015 0.030",

            "rgba":
                "0.85 0.45 0.15 1",

            "contype":
                str(OBJECT_TYPE),

            # OBJECT collides with:
            # HAND(bit 1) + TABLE(bit 4)
            "conaffinity":
                str(
                    HAND_TYPE
                    | TABLE_TYPE
                ),

            "condim":
                "3",

            "friction":
                "0.9 0.01 0.001",
        }
    )

    print(
        "      radius      = 15 mm"
    )

    print(
        "      half-length = 30 mm"
    )


def save_model(
    root: ET.Element,
):
    print(
        "[4/4] Saving contact model..."
    )

    ET.indent(
        root,
        space="  "
    )

    ET.ElementTree(
        root
    ).write(
        OUTPUT_PATH,
        encoding="utf-8",
        xml_declaration=True,
    )


def validate():
    print(
        "\n[VALIDATE]"
    )

    model = (
        mujoco.MjModel
        .from_xml_path(
            str(OUTPUT_PATH)
        )
    )

    print(
        "========================================"
    )

    print(
        "[CONTACT MODEL]"
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
        "========================================"
    )

    # Original 42 hand geoms
    # + table
    # + cylinder
    if model.ngeom != 44:

        raise RuntimeError(
            f"Expected ngeom=44, "
            f"got {model.ngeom}"
        )

    if model.nmocap != 1:

        raise RuntimeError(
            f"Expected nmocap=1, "
            f"got {model.nmocap}"
        )

    if model.nu != 20:

        raise RuntimeError(
            f"Expected nu=20, "
            f"got {model.nu}"
        )

    if model.neq != 20:

        raise RuntimeError(
            f"Expected neq=20, "
            f"got {model.neq}"
        )

    # Important geometry names
    required = (
        "right_index_distal_geom",
        "right_index_middle_geom",
        "contact_cylinder_geom",
        "contact_test_table_geom",
    )

    print(
        "\n[IMPORTANT GEOMS]"
    )

    for name in required:

        gid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        print(
            f"  {name:<32s}"
            f" id={gid}"
        )

        if gid < 0:

            raise RuntimeError(
                f"Missing geom: {name}"
            )

    print(
        "\n[SUCCESS]"
    )

    print(
        "Contact model saved to:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )


def main():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"{INPUT_PATH} not found.\n"
            "Run build_l20lite_dynamic.py first."
        )

    print(
        "========================================"
    )

    print(
        " L20 Lite Contact Model Builder"
    )

    print(
        "========================================"
    )

    print(
        "Input:"
    )

    print(
        f"  {INPUT_PATH}"
    )

    root = (
        ET.parse(
            INPUT_PATH
        ).getroot()
    )

    remove_old_test_scene(
        root
    )

    configure_hand_contact(
        root
    )

    add_table(
        root
    )

    add_mocap_cylinder(
        root
    )

    save_model(
        root
    )

    validate()


if __name__ == "__main__":
    main()
