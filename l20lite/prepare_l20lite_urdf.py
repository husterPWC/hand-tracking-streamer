#!/usr/bin/env python3
from __future__ import annotations

import copy
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent

LEFT_URDF = "linkerhand_l20lite_left.urdf"
RIGHT_URDF = "linkerhand_l20lite_right.urdf"


def find_unique(root: Path, filename: str) -> Path:
    matches = [
        p for p in root.rglob(filename)
        if "generated" not in p.parts
    ]

    if not matches:
        raise FileNotFoundError(
            f"Cannot find {filename} under {root}"
        )

    if len(matches) > 1:
        print(
            f"[WARN] Found {len(matches)} copies of "
            f"{filename}; using {matches[0]}"
        )

    return matches[0]


def read_repaired_urdf(path: Path) -> str:
    text = path.read_text(
        encoding="utf-8-sig"
    )

    # Normalize Windows CRLF.
    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    # The supplied LEFT URDF is missing "<robot".
    head = text[:500]

    if re.search(r"<robot(?:\s|>)", head) is None:

        lines = text.splitlines()

        if (
            not lines
            or not lines[0]
            .lstrip()
            .startswith("<?xml")
        ):
            raise ValueError(
                f"{path}: missing <robot> "
                "and XML header is unexpected"
            )

        lines.insert(
            1,
            "<robot"
        )

        text = "\n".join(lines)

        if not text.endswith("\n"):
            text += "\n"

        print(
            f"[FIX] Repaired missing "
            f"<robot> token: {path}"
        )

    # XML syntax validation.
    ET.fromstring(text)

    return text


def write_fixed_copy(
    src: Path,
    text: str,
) -> Path:

    out = src.with_name(
        src.stem + "_fixed.urdf"
    )

    with out.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write(text)

    return out


def summarize(
    root: ET.Element,
    side: str,
) -> None:

    joints = root.findall("joint")

    active = [
        j
        for j in joints
        if (
            j.attrib.get("type") != "fixed"
            and j.find("mimic") is None
        )
    ]

    mimic = [
        j
        for j in joints
        if j.find("mimic") is not None
    ]

    print(
        f"\n[{side.upper()}] "
        f"links={len(root.findall('link'))}, "
        f"joints={len(joints)}, "
        f"active={len(active)}, "
        f"mimic={len(mimic)}"
    )

    print("  Active joints:")

    for i, j in enumerate(active):

        lim = j.find("limit")

        lo = (
            lim.attrib.get("lower", "?")
            if lim is not None
            else "?"
        )

        hi = (
            lim.attrib.get("upper", "?")
            if lim is not None
            else "?"
        )

        print(
            f"    {i:2d}: "
            f"{j.attrib['name']:<24s} "
            f"[{lo}, {hi}]"
        )

    print("  Mimic joints:")

    for j in mimic:

        m = j.find("mimic")

        print(
            f"      "
            f"{j.attrib['name']:<24s} "
            f"<- {m.attrib['joint']} "
            f"* {m.attrib.get('multiplier', '1')} "
            f"+ {m.attrib.get('offset', '0')}"
        )


def prefix_hand_into_dual(
    dual_root: ET.Element,
    hand_root: ET.Element,
    side: str,
    source_dir: Path,
    generated_dir: Path,
) -> str:

    link_map = {
        link.attrib["name"]:
        f"{side}_{link.attrib['name']}"
        for link in hand_root.findall("link")
    }

    joint_map = {
        joint.attrib["name"]:
        f"{side}_{joint.attrib['name']}"
        for joint in hand_root.findall("joint")
    }

    mesh_dir = generated_dir

    mesh_cache = {}

    # ------------------------------
    # Links
    # ------------------------------
    for link in hand_root.findall("link"):

        node = copy.deepcopy(link)

        node.attrib["name"] = (
            link_map[
                link.attrib["name"]
            ]
        )

        for mesh in node.findall(".//mesh"):

            raw = mesh.attrib.get(
                "filename",
                ""
            )

            raw_path = raw

            if raw_path.startswith("file://"):
                raw_path = raw_path[
                    len("file://"):
                ]

            if raw_path.startswith(
                "package://"
            ):
                raise ValueError(
                    "package:// mesh URI "
                    "is not handled automatically: "
                    f"{raw}"
                )

            src_mesh = Path(raw_path)

            if not src_mesh.is_absolute():

                src_mesh = (
                    source_dir
                    / src_mesh
                ).resolve()

            if not src_mesh.exists():
                raise FileNotFoundError(
                    f"Missing mesh: "
                    f"{src_mesh}"
                )

            if src_mesh not in mesh_cache:

                dst_name = (
                    f"{side}_"
                    f"{src_mesh.name}"
                )

                dst_mesh = (
                    mesh_dir
                    / dst_name
                )

                shutil.copy2(
                    src_mesh,
                    dst_mesh,
                )

                mesh_cache[src_mesh] = dst_name

            mesh.attrib["filename"] = (
                mesh_cache[src_mesh]
            )

        dual_root.append(node)

    # ------------------------------
    # Joints
    # ------------------------------
    for joint in hand_root.findall("joint"):

        node = copy.deepcopy(joint)

        old_name = joint.attrib["name"]

        node.attrib["name"] = (
            joint_map[old_name]
        )

        parent = node.find("parent")
        child = node.find("child")

        parent.attrib["link"] = (
            link_map[
                parent.attrib["link"]
            ]
        )

        child.attrib["link"] = (
            link_map[
                child.attrib["link"]
            ]
        )

        mimic = node.find("mimic")

        if mimic is not None:

            mimic.attrib["joint"] = (
                joint_map[
                    mimic.attrib["joint"]
                ]
            )

        dual_root.append(node)

    return link_map[
        "hand_base_link"
    ]


def add_mount(
    dual_root: ET.Element,
    side: str,
    child_link: str,
    xyz,
):

    joint = ET.SubElement(
        dual_root,
        "joint",
        {
            "name": f"{side}_mount",
            "type": "fixed",
        }
    )

    ET.SubElement(
        joint,
        "origin",
        {
            "xyz":
                f"{xyz[0]} "
                f"{xyz[1]} "
                f"{xyz[2]}",
            "rpy": "0 0 0",
        }
    )

    ET.SubElement(
        joint,
        "parent",
        {
            "link": "world"
        }
    )

    ET.SubElement(
        joint,
        "child",
        {
            "link": child_link
        }
    )


def main():

    left_src = find_unique(
        ROOT,
        LEFT_URDF
    )

    right_src = find_unique(
        ROOT,
        RIGHT_URDF
    )

    print("[FOUND]")
    print(" left :", left_src)
    print(" right:", right_src)

    # ------------------------------
    # Repair
    # ------------------------------
    left_text = read_repaired_urdf(
        left_src
    )

    right_text = read_repaired_urdf(
        right_src
    )

    left_fixed = write_fixed_copy(
        left_src,
        left_text,
    )

    right_fixed = write_fixed_copy(
        right_src,
        right_text,
    )

    print("\n[FIXED COPIES]")
    print(" left :", left_fixed)
    print(" right:", right_fixed)

    left_root = ET.fromstring(
        left_text
    )

    right_root = ET.fromstring(
        right_text
    )

    summarize(
        left_root,
        "left",
    )

    summarize(
        right_root,
        "right",
    )

    # ------------------------------
    # Generate dual hand URDF
    # ------------------------------
    generated = (
        ROOT
        / "generated"
    )

    if generated.exists():
        shutil.rmtree(
            generated
        )

    generated.mkdir(
        parents=True
    )

    dual_root = ET.Element(
        "robot",
        {
            "name":
            "linkerhand_l20lite_dual"
        }
    )

    # Artificial common root.
    ET.SubElement(
        dual_root,
        "link",
        {
            "name": "world"
        }
    )

    left_base = prefix_hand_into_dual(
        dual_root,
        left_root,
        "left",
        left_src.parent,
        generated,
    )

    right_base = prefix_hand_into_dual(
        dual_root,
        right_root,
        "right",
        right_src.parent,
        generated,
    )

    # 左右手分开 23 cm。
    add_mount(
        dual_root,
        "left",
        left_base,
        (
            0.0,
            0.115,
            0.0,
        )
    )

    add_mount(
        dual_root,
        "right",
        right_base,
        (
            0.0,
            -0.115,
            0.0,
        )
    )

    ET.indent(
        dual_root,
        space="  "
    )

    dual_path = (
        generated
        / "dual_l20lite.urdf"
    )

    ET.ElementTree(
        dual_root
    ).write(
        dual_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    # final XML check
    ET.parse(
        dual_path
    )

    print("\n[DONE]")
    print(
        " dual URDF:",
        dual_path,
    )

    print(
        " copied meshes:",
        len(
            list(
                generated.glob("*.STL")
            )
        )
    )

    print(
        "\nNext:"
        "\n  python "
        "check_mujoco_load.py"
    )


if __name__ == "__main__":
    main()
