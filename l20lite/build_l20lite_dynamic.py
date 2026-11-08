#!/usr/bin/env python3
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent

URDF_PATH = (
    ROOT
    / "generated"
    / "dual_l20lite.urdf"
)

OUTPUT_PATH = (
    ROOT
    / "generated"
    / "dual_l20lite_dynamic.xml"
)


# ============================================================
# L20 Lite SDK 10-D control order
# ============================================================

ACTIVE_JOINTS = (
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


def parse_urdf_joint_info(urdf_path: Path):
    """
    Read:
      1. joint limits
      2. mimic relations

    mimic:
        child = multiplier * parent + offset
    """

    root = ET.parse(
        urdf_path
    ).getroot()

    limits = {}
    mimic_relations = []

    for joint in root.findall("joint"):

        name = joint.attrib["name"]

        # ----------------------------
        # limits
        # ----------------------------
        limit = joint.find("limit")

        if limit is not None:

            lower = float(
                limit.attrib.get(
                    "lower",
                    "0",
                )
            )

            upper = float(
                limit.attrib.get(
                    "upper",
                    "0",
                )
            )

            limits[name] = (
                lower,
                upper,
            )

        # ----------------------------
        # mimic
        # ----------------------------
        mimic = joint.find("mimic")

        if mimic is not None:

            parent = mimic.attrib[
                "joint"
            ]

            multiplier = float(
                mimic.attrib.get(
                    "multiplier",
                    "1",
                )
            )

            offset = float(
                mimic.attrib.get(
                    "offset",
                    "0",
                )
            )

            mimic_relations.append(
                (
                    name,
                    parent,
                    multiplier,
                    offset,
                )
            )

    return (
        limits,
        mimic_relations,
    )


def convert_urdf_to_mjcf(
    urdf_path: Path,
):
    """
    MuJoCo 3.3.x can parse URDF into MjSpec.

    MjSpec.to_xml() serializes it as MJCF.
    """

    print(
        "[1/4] Parsing URDF with MjSpec..."
    )

    spec = (
        mujoco.MjSpec
        .from_file(
            str(urdf_path)
        )
    )

    # Compile once to make sure the URDF is valid.
    model = spec.compile()

    print(
        "      URDF compiled:"
        f" nq={model.nq},"
        f" nv={model.nv},"
        f" nu={model.nu}"
    )

    mjcf_text = spec.to_xml()

    return ET.fromstring(
        mjcf_text
    )


def configure_simulation(
    root: ET.Element,
):
    """
    Configure MuJoCo dynamic simulation.
    """

    print(
        "[2/4] Configuring simulation..."
    )

    option = root.find("option")

    if option is None:

        option = ET.Element(
            "option"
        )

        # Put option near top of MJCF.
        root.insert(
            1,
            option,
        )

    option.attrib[
        "timestep"
    ] = "0.002"

    option.attrib[
        "gravity"
    ] = "0 0 -9.81"

    # Position actuator with damping behaves
    # better with implicit integration.
    option.attrib[
        "integrator"
    ] = "implicitfast"

def _subtree_has_joint_prefix(
    body: ET.Element,
    prefix: str,
) -> bool:
    """
    Check whether this body subtree contains
    any joint whose name begins with prefix.
    """

    for joint in body.iter("joint"):

        name = joint.attrib.get(
            "name",
            ""
        )

        if name.startswith(prefix):
            return True

    return False


def configure_hand_bodies(
    root: ET.Element,
):
    """
    Handle both:

      1. articulated finger/root body trees
      2. palm/base geoms merged directly into worldbody

    after URDF -> MJCF conversion.

    This stage temporarily disables all hand collisions
    so that actuator + equality dynamics can be verified
    independently.
    """

    print(
        "[2.5/4] Configuring hand bodies..."
    )

    worldbody = root.find(
        "worldbody"
    )

    if worldbody is None:
        raise RuntimeError(
            "MJCF has no <worldbody>"
        )

    # ========================================================
    # 1. Find all articulated top-level hand body trees
    # ========================================================

    top_bodies = worldbody.findall(
        "body"
    )

    left_roots = []
    right_roots = []

    for body in top_bodies:

        if _subtree_has_joint_prefix(
            body,
            "left_",
        ):
            left_roots.append(
                body
            )

        elif _subtree_has_joint_prefix(
            body,
            "right_",
        ):
            right_roots.append(
                body
            )

    if not left_roots:
        raise RuntimeError(
            "Cannot find any left-hand "
            "top-level body trees"
        )

    if not right_roots:
        raise RuntimeError(
            "Cannot find any right-hand "
            "top-level body trees"
        )

    print(
        f"      detected LEFT root trees : "
        f"{len(left_roots)}"
    )

    for body in left_roots:
        print(
            "        ",
            body.attrib.get(
                "name",
                "<unnamed>",
            )
        )

    print(
        f"      detected RIGHT root trees: "
        f"{len(right_roots)}"
    )

    for body in right_roots:
        print(
            "        ",
            body.attrib.get(
                "name",
                "<unnamed>",
            )
        )

    # ========================================================
    # Helper: add Y translation while preserving X/Z
    # ========================================================

    def add_y_offset(
        element: ET.Element,
        offset: float,
    ):

        raw = element.attrib.get(
            "pos",
            "0 0 0",
        )

        values = [
            float(v)
            for v in raw.split()
        ]

        if len(values) != 3:
            values = [
                0.0,
                0.0,
                0.0,
            ]

        values[1] += offset

        element.attrib["pos"] = (
            f"{values[0]:.8g} "
            f"{values[1]:.8g} "
            f"{values[2]:.8g}"
        )

    # ========================================================
    # 2. Move articulated hand branches
    # ========================================================

    for body in left_roots:
        add_y_offset(
            body,
            +0.18,
        )

    for body in right_roots:
        add_y_offset(
            body,
            -0.18,
        )

    print(
        "      applied LEFT body offset : "
        "+0.18 m in y"
    )

    print(
        "      applied RIGHT body offset: "
        "-0.18 m in y"
    )

    # ========================================================
    # 3. Disable collision for articulated branches
    # ========================================================

    left_geom_count = 0
    right_geom_count = 0

    for body in left_roots:

        for geom in body.iter(
            "geom"
        ):

            geom.attrib[
                "contype"
            ] = "0"

            geom.attrib[
                "conaffinity"
            ] = "0"

            left_geom_count += 1

    for body in right_roots:

        for geom in body.iter(
            "geom"
        ):

            geom.attrib[
                "contype"
            ] = "0"

            geom.attrib[
                "conaffinity"
            ] = "0"

            right_geom_count += 1

    # ========================================================
    # 4. Handle palm/base geoms merged into worldbody
    # ========================================================

    world_geoms = worldbody.findall(
        "geom"
    )

    left_world_geoms = []
    right_world_geoms = []
    unknown_world_geoms = []

    for geom in world_geoms:

        geom_name = geom.attrib.get(
            "name",
            ""
        )

        mesh_name = geom.attrib.get(
            "mesh",
            ""
        )

        signature = (
            geom_name
            + " "
            + mesh_name
        ).lower()

        if "left_" in signature:

            left_world_geoms.append(
                geom
            )

        elif "right_" in signature:

            right_world_geoms.append(
                geom
            )

        else:

            unknown_world_geoms.append(
                geom
            )

    print(
        f"      world-level LEFT geoms : "
        f"{len(left_world_geoms)}"
    )

    for geom in left_world_geoms:

        print(
            "        ",
            "name=",
            geom.attrib.get(
                "name",
                "<unnamed>",
            ),
            "mesh=",
            geom.attrib.get(
                "mesh",
                "<none>",
            ),
        )

        add_y_offset(
            geom,
            +0.18,
        )

        geom.attrib[
            "contype"
        ] = "0"

        geom.attrib[
            "conaffinity"
        ] = "0"

        left_geom_count += 1

    print(
        f"      world-level RIGHT geoms: "
        f"{len(right_world_geoms)}"
    )

    for geom in right_world_geoms:

        print(
            "        ",
            "name=",
            geom.attrib.get(
                "name",
                "<unnamed>",
            ),
            "mesh=",
            geom.attrib.get(
                "mesh",
                "<none>",
            ),
        )

        add_y_offset(
            geom,
            -0.18,
        )

        geom.attrib[
            "contype"
        ] = "0"

        geom.attrib[
            "conaffinity"
        ] = "0"

        right_geom_count += 1

    # ========================================================
    # Debug unknown world-level geometry
    # ========================================================

    if unknown_world_geoms:

        print(
            "      [WARN] Unclassified "
            "world-level geoms:"
        )

        for geom in unknown_world_geoms:

            print(
                "        name=",
                geom.attrib.get(
                    "name",
                    "<unnamed>",
                ),
                "mesh=",
                geom.attrib.get(
                    "mesh",
                    "<none>",
                ),
            )

    # ========================================================
    # Summary
    # ========================================================

    total = (
        left_geom_count
        + right_geom_count
    )

    print(
        "      collision disabled:"
    )

    print(
        f"        left geoms : "
        f"{left_geom_count}"
    )

    print(
        f"        right geoms: "
        f"{right_geom_count}"
    )

    print(
        f"        total       : "
        f"{total}"
    )


def configure_joint_dynamics(
    root: ET.Element,
):
    """
    Add modest passive damping and armature.

    These are simulation stabilization parameters,
    NOT claimed to be physical L20 Lite parameters.
    """

    print(
        "[2.6/4] Adding joint damping..."
    )

    count = 0

    for body in root.iter(
        "body"
    ):

        for joint in body.findall(
            "joint"
        ):

            joint.attrib[
                "damping"
            ] = "0.02"

            joint.attrib[
                "armature"
            ] = "0.0002"

            count += 1

    print(
        f"      configured "
        f"{count} joints"
    )




def add_mimic_equalities(
    root: ET.Element,
    mimic_relations,
):
    """
    URDF mimic:
        q_child = multiplier * q_parent + offset

    MuJoCo equality/joint:
        joint1 = polynomial(joint2)

    polycoef:
        a0 a1 a2 a3 a4

    Therefore:
        a0 = offset
        a1 = multiplier
    """

    print(
        "[3/4] Adding mimic equality constraints..."
    )

    # Remove existing equality section
    # if generated MJCF happens to contain one.
    old = root.find(
        "equality"
    )

    if old is not None:
        root.remove(
            old
        )

    equality = ET.SubElement(
        root,
        "equality",
    )

    for (
        child,
        parent,
        multiplier,
        offset,
    ) in mimic_relations:

        ET.SubElement(
            equality,
            "joint",
            {
                "name":
                    f"mimic_{child}",

                "joint1":
                    child,

                "joint2":
                    parent,

                "polycoef":
                    (
                        f"{offset:.10g} "
                        f"{multiplier:.10g} "
                        "0 0 0"
                    ),

                # Slightly stiffer than default.
                # timestep = 0.002 s
                "solref":
                    "0.01 1",

                "solimp":
                    "0.95 0.99 0.001",
            }
        )

        print(
            f"      {child:<32s}"
            f" <- {parent:<32s}"
            f" * {multiplier:.4f}"
            f" + {offset:.4f}"
        )


def add_actuators(
    root: ET.Element,
    limits,
):
    """
    Add 10 position actuators per hand.

    Total:
        20 actuators.
    """

    print(
        "[4/4] Adding position actuators..."
    )

    old = root.find(
        "actuator"
    )

    if old is not None:
        root.remove(
            old
        )

    actuator = ET.SubElement(
        root,
        "actuator",
    )

    actuator_count = 0

    for side in (
        "left",
        "right",
    ):

        print(
            f"\n      [{side.upper()}]"
        )

        for index, short_name in enumerate(
            ACTIVE_JOINTS
        ):

            joint_name = (
                f"{side}_{short_name}"
            )

            if joint_name not in limits:

                raise KeyError(
                    "Missing joint limit for "
                    f"{joint_name}"
                )

            lower, upper = (
                limits[
                    joint_name
                ]
            )

            actuator_name = (
                f"{side}_a{index}_"
                f"{short_name}"
            )

            ET.SubElement(
                actuator,
                "position",
                {
                    "name":
                        actuator_name,

                    "joint":
                        joint_name,

                    # Initial control range exactly
                    # follows URDF joint range.
                    "ctrllimited":
                        "true",

                    "ctrlrange":
                        (
                            f"{lower:.10g} "
                            f"{upper:.10g}"
                        ),

                    # Initial gains.
                    # These are simulation values,
                    # NOT claimed to be real L20 Lite
                    # motor gains.
                    "kp":
                        "5",

                    # Critical damping based on
                    # MuJoCo's effective mass estimate.
                    "dampratio":
                        "1.5",
                }
            )

            actuator_count += 1

            print(
                f"        ctrl[{actuator_count - 1:2d}] "
                f"{joint_name:<30s} "
                f"[{lower:.3f}, "
                f"{upper:.3f}]"
            )

    return actuator_count


def validate_model(
    output_path: Path,
    expected_actuators: int,
    expected_equalities: int,
):

    print(
        "\n[VALIDATE] Loading generated MJCF..."
    )

    model = (
        mujoco.MjModel
        .from_xml_path(
            str(output_path)
        )
    )

    print(
        "\n========================================"
    )

    print(
        "[DYNAMIC MODEL]"
    )

    print(
        f"nq    = {model.nq}"
    )

    print(
        f"nv    = {model.nv}"
    )

    print(
        f"nu    = {model.nu}"
    )

    print(
        f"njnt  = {model.njnt}"
    )

    print(
        f"neq   = {model.neq}"
    )

    print(
        f"nbody = {model.nbody}"
    )

    print(
        f"ngeom = {model.ngeom}"
    )

    print(
        "========================================"
    )

    if model.nu != expected_actuators:

        raise RuntimeError(
            f"Expected nu="
            f"{expected_actuators}, "
            f"but got {model.nu}"
        )

    if model.neq != expected_equalities:

        raise RuntimeError(
            f"Expected neq="
            f"{expected_equalities}, "
            f"but got {model.neq}"
        )

    print(
        "\n[ACTUATORS]"
    )

    for aid in range(
        model.nu
    ):

        name = (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                aid,
            )
        )

        lo, hi = (
            model.actuator_ctrlrange[
                aid
            ]
        )

        print(
            f"{aid:2d}: "
            f"{name:<45s} "
            f"ctrl=[{lo:.4f}, "
            f"{hi:.4f}]"
        )

    print(
        "\n[SUCCESS]"
    )

    print(
        "Dynamic L20 Lite model is ready."
    )

    print(
        f"Saved to:\n  {output_path}"
    )


def main():

    if not URDF_PATH.exists():

        raise FileNotFoundError(
            f"{URDF_PATH} does not exist"
        )

    print(
        "========================================"
    )

    print(
        " L20 Lite Dynamic Model Builder"
    )

    print(
        "========================================"
    )

    print(
        "Input:"
    )

    print(
        f"  {URDF_PATH}"
    )

    # ----------------------------------
    # Read URDF-specific information
    # ----------------------------------

    (
        limits,
        mimic_relations,
    ) = parse_urdf_joint_info(
        URDF_PATH
    )

    print(
        f"\nFound "
        f"{len(limits)} joint limits"
    )

    print(
        f"Found "
        f"{len(mimic_relations)} "
        f"mimic joints"
    )

    if len(
        mimic_relations
    ) != 20:

        raise RuntimeError(
            "Expected 20 mimic relations "
            "for dual L20 Lite, got "
            f"{len(mimic_relations)}"
        )

    # ----------------------------------
    # URDF -> MJCF
    # ----------------------------------

    root = convert_urdf_to_mjcf(
        URDF_PATH
    )

    configure_simulation(
    root
    )

    configure_hand_bodies(
        root
    )

    configure_joint_dynamics(
        root
    )

    add_mimic_equalities(
        root,
        mimic_relations,
    )

    num_actuators = (
        add_actuators(
            root,
            limits,
        )
    )

    # ----------------------------------
    # Save
    # ----------------------------------

    ET.indent(
        root,
        space="  "
    )

    tree = ET.ElementTree(
        root
    )

    tree.write(
        OUTPUT_PATH,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(
        "\nSaved preliminary MJCF:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    # ----------------------------------
    # Validate
    # ----------------------------------

    validate_model(
        OUTPUT_PATH,
        expected_actuators=num_actuators,
        expected_equalities=len(
            mimic_relations
        ),
    )


if __name__ == "__main__":
    main()
