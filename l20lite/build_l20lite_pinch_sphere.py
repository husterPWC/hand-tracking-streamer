#!/usr/bin/env python3
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent

INPUT_PATH = (
    ROOT
    / "generated"
    / "dual_l20lite_tactile.xml"
)

OUTPUT_PATH = (
    ROOT
    / "generated"
    / "dual_l20lite_pinch_sphere.xml"
)


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

        # Give deterministic names.
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

        geom.attrib["friction"] = (
            "1.2 0.01 0.001"
        )

    print(
        f"      left  geoms = {left_count}"
    )

    print(
        f"      right geoms = {right_count}"
    )

    if (
        left_count != 21
        or right_count != 21
    ):

        raise RuntimeError(
            "Expected 21 geoms per hand, "
            f"got left={left_count}, "
            f"right={right_count}"
        )


def add_target(
    root: ET.Element,
):

    print(
        "[2/4] Adding mocap target..."
    )

    worldbody = root.find(
        "worldbody"
    )

    if worldbody is None:

        raise RuntimeError(
            "Missing worldbody"
        )

    ET.SubElement(
        worldbody,
        "body",
        {
            "name":
                "pinch_target_mocap",

            "mocap":
                "true",

            "pos":
                "0 -0.18 0.18",
        },
    )


def add_sphere(
    root: ET.Element,
):

    print(
        "[3/4] Adding free sphere..."
    )

    worldbody = root.find(
        "worldbody"
    )

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name":
                "pinch_sphere",

            "pos":
                "0 -0.18 0.18",
        },
    )

    ET.SubElement(
        body,
        "freejoint",
        {
            "name":
                "pinch_sphere_free",
        },
    )

    ET.SubElement(
        body,
        "geom",
        {
            "name":
                "pinch_sphere_geom",

            "type":
                "sphere",

            # diameter = 24 mm
            "size":
                "0.012",

            "mass":
                "0.030",

            "rgba":
                "0.90 0.40 0.10 1",

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

            "solref":
                "0.01 1",

            "solimp":
                "0.90 0.95 0.001",
        },
    )


def add_weld(
    root: ET.Element,
):

    print(
        "[4/4] Adding temporary weld..."
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
                "pinch_sphere_weld",

            "body1":
                "pinch_sphere",

            "body2":
                "pinch_target_mocap",

            "solref":
                "0.005 1",

            "solimp":
                "0.95 0.99 0.001",
        },
    )


def validate():

    model = (
        mujoco.MjModel
        .from_xml_path(
            str(
                OUTPUT_PATH
            )
        )
    )

    print(
        "\n========================================"
    )

    print(
        "[PINCH SPHERE MODEL]"
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
        f"nmocap  = {model.nmocap}"
    )

    print(
        "========================================"
    )

    # Original:
    # nq=40 nv=40
    #
    # free sphere:
    # +7 qpos, +6 qvel
    if model.nq != 47:

        raise RuntimeError(
            f"Expected nq=47, got {model.nq}"
        )

    if model.nv != 46:

        raise RuntimeError(
            f"Expected nv=46, got {model.nv}"
        )

    if model.nu != 20:

        raise RuntimeError(
            f"Expected nu=20, got {model.nu}"
        )

    # 20 mimic + 1 weld
    if model.neq != 21:

        raise RuntimeError(
            f"Expected neq=21, got {model.neq}"
        )

    # 42 hand + 1 sphere
    if model.ngeom != 43:

        raise RuntimeError(
            f"Expected ngeom=43, got {model.ngeom}"
        )

    if model.nmocap != 1:

        raise RuntimeError(
            f"Expected nmocap=1, "
            f"got {model.nmocap}"
        )

    required_sites = (
        "right_thumb_tip_site",
        "right_index_tip_site",
    )

    for name in required_sites:

        sid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )

        if sid < 0:

            raise RuntimeError(
                f"Missing site: {name}"
            )

    print(
        "\n[SUCCESS]"
    )

    print(
        f"Saved:\n  {OUTPUT_PATH}"
    )


def main():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"{INPUT_PATH}\n"
            "Run build_l20lite_tactile.py first."
        )

    print(
        "========================================"
    )

    print(
        " L20 Lite Pinch Sphere Builder"
    )

    print(
        "========================================"
    )

    root = (
        ET.parse(
            INPUT_PATH
        ).getroot()
    )

    configure_hand_contact(
        root
    )

    add_target(
        root
    )

    add_sphere(
        root
    )

    add_weld(
        root
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
