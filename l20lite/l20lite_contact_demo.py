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
    / "dual_l20lite_contact.xml"
)


# ============================================================
# Commands
# ============================================================

OPEN = np.zeros(
    10,
    dtype=np.float64,
)


# Keep the index slightly flexed.
# This makes its distal link easier to inspect.
CONTACT_POSE = np.array(
    [
        0,      # thumb flex
        0,      # thumb yaw
        35,     # index flex
        0,      # middle
        0,      # ring
        0,      # pinky
        0,      # index side
        0,
        0,
        0,
    ],
    dtype=np.float64,
)


# ============================================================

def geom_id(
    model,
    name,
):

    gid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        name,
    )

    if gid < 0:
        raise KeyError(
            f"Geom not found: {name}"
        )

    return gid


def body_id(
    model,
    name,
):

    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    if bid < 0:
        raise KeyError(
            f"Body not found: {name}"
        )

    return bid


# ============================================================
# Contact reading
# ============================================================

def classify_finger(
    geom_name: str,
):

    for finger in (
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    ):

        if (
            f"_{finger}_"
            in geom_name
        ):

            return finger

    if "hand_base" in geom_name:
        return "palm"

    return "other"


def read_cylinder_contacts(
    sim,
    cylinder_gid,
):

    """
    Returns only contacts involving the test cylinder.

    mj_contactForce gives:
        [Fx, Fy, Fz, Tx, Ty, Tz]

    in the CONTACT frame.

    Contact-frame X is the contact normal direction,
    therefore abs(force[0]) is used as the compressive
    normal-force magnitude for this diagnostic.
    """

    model = sim.model
    data = sim.data

    total_normal = 0.0
    total_tangent = 0.0

    per_finger = {}

    contact_rows = []

    for cid in range(
        data.ncon
    ):

        contact = data.contact[
            cid
        ]

        g1 = int(
            contact.geom1
        )

        g2 = int(
            contact.geom2
        )

        if (
            g1 != cylinder_gid
            and g2 != cylinder_gid
        ):
            continue

        other_gid = (
            g2
            if g1 == cylinder_gid
            else g1
        )

        other_name = (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                other_gid,
            )
        )

        if other_name is None:
            other_name = (
                f"geom_{other_gid}"
            )

        # Only interested in hand contacts.
        if not (
            other_name.startswith(
                "left_"
            )
            or other_name.startswith(
                "right_"
            )
        ):
            continue

        force6 = np.zeros(
            6,
            dtype=np.float64,
        )

        mujoco.mj_contactForce(
            model,
            data,
            cid,
            force6,
        )

        normal = abs(
            float(
                force6[0]
            )
        )

        tangential = float(
            np.linalg.norm(
                force6[1:3]
            )
        )

        finger = classify_finger(
            other_name
        )

        if finger not in per_finger:

            per_finger[finger] = {
                "normal": 0.0,
                "tangent": 0.0,
                "count": 0,
            }

        per_finger[
            finger
        ]["normal"] += normal

        per_finger[
            finger
        ]["tangent"] += tangential

        per_finger[
            finger
        ]["count"] += 1

        total_normal += normal
        total_tangent += tangential

        contact_rows.append(
            {
                "finger":
                    finger,

                "geom":
                    other_name,

                "normal":
                    normal,

                "tangent":
                    tangential,

                "pos":
                    np.array(
                        contact.pos,
                        dtype=np.float64,
                    ),
            }
        )

    return {
        "normal":
            total_normal,

        "tangent":
            total_tangent,

        "per_finger":
            per_finger,

        "contacts":
            contact_rows,
    }


# ============================================================
# Timing helper
# ============================================================

def realtime_step(
    sim,
):

    start = time.perf_counter()

    sim.step()

    elapsed = (
        time.perf_counter()
        - start
    )

    sleep_time = (
        sim.timestep
        - elapsed
    )

    if sleep_time > 0:

        time.sleep(
            sleep_time
        )


def hold(
    sim,
    viewer,
    seconds,
):

    end_time = (
        sim.data.time
        + seconds
    )

    count = 0

    while (
        viewer.is_running()
        and sim.data.time < end_time
    ):

        realtime_step(
            sim
        )

        count += 1

        if count % 10 == 0:
            viewer.sync()


# ============================================================
# Main
# ============================================================

def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"{MODEL_PATH} not found.\n"
            "Run build_l20lite_contact.py first."
        )

    sim = L20LiteMuJoCo(
        MODEL_PATH
    )

    model = sim.model
    data = sim.data

    cylinder_gid = geom_id(
        model,
        "contact_cylinder_geom",
    )

    index_distal_gid = geom_id(
        model,
        "right_index_distal_geom",
    )

    index_middle_gid = geom_id(
        model,
        "right_index_middle_geom",
    )

    cylinder_bid = body_id(
        model,
        "contact_cylinder_mocap",
    )

    mocap_id = int(
        model.body_mocapid[
            cylinder_bid
        ]
    )

    if mocap_id < 0:

        raise RuntimeError(
            "Cylinder body is not mocap"
        )

    print(
        "================================="
    )

    print(
        " L20 Lite Contact Force Test"
    )

    print(
        "================================="
    )

    print(
        f"nq={model.nq}, "
        f"nu={model.nu}, "
        f"neq={model.neq}, "
        f"ngeom={model.ngeom}"
    )

    print(
        f"mocap_id = {mocap_id}"
    )

    # --------------------------------------------
    # Set hand pose
    # --------------------------------------------

    sim.left.angle.set_angles(
        OPEN
    )

    sim.right.angle.set_angles(
        CONTACT_POSE
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # Optional contact visualization
        try:

            viewer.opt.flags[
                mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
            ] = True

            viewer.opt.flags[
                mujoco.mjtVisFlag.mjVIS_CONTACTFORCE
            ] = True

        except Exception:
            pass

        viewer.cam.lookat[:] = np.array(
            [
                0.0,
                -0.18,
                0.10,
            ]
        )

        viewer.cam.distance = 0.32
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -18

        viewer.sync()

        # ----------------------------------------
        # Let hand settle first.
        # ----------------------------------------

        print(
            "\n[1] Settling right hand..."
        )

        hold(
            sim,
            viewer,
            1.5,
        )

        mujoco.mj_forward(
            model,
            data,
        )

        # ----------------------------------------
        # Determine index finger direction
        #
        # Use:
        #   index_middle center -> index_distal center
        #
        # as approximate outward finger direction.
        # ----------------------------------------

        distal_pos = np.array(
            data.geom_xpos[
                index_distal_gid
            ],
            dtype=np.float64,
        )

        middle_pos = np.array(
            data.geom_xpos[
                index_middle_gid
            ],
            dtype=np.float64,
        )

        outward = (
            distal_pos
            - middle_pos
        )

        norm = np.linalg.norm(
            outward
        )

        if norm < 1e-6:

            outward = np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                ]
            )

        else:

            outward = (
                outward
                / norm
            )

        print(
            "\n[Index geometry]"
        )

        print(
            " middle :",
            np.round(
                middle_pos,
                4,
            ).tolist()
        )

        print(
            " distal :",
            np.round(
                distal_pos,
                4,
            ).tolist()
        )

        print(
            " outward:",
            np.round(
                outward,
                4,
            ).tolist()
        )

        # ----------------------------------------
        # Cylinder starts 8 cm outside fingertip.
        # ----------------------------------------

        start_pos = (
            distal_pos
            + outward * 0.08
        )

        # Final geometric target passes into the
        # fingertip center. We normally stop earlier
        # as soon as contact is detected.
        deep_target = (
            distal_pos
            - outward * 0.01
        )

        data.mocap_pos[
            mocap_id
        ] = start_pos

        data.mocap_quat[
            mocap_id
        ] = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        mujoco.mj_forward(
            model,
            data,
        )

        viewer.sync()

        print(
            "\n[2] Cylinder positioned outside "
            "right index fingertip."
        )

        print(
            " start:",
            np.round(
                start_pos,
                4,
            ).tolist()
        )

        hold(
            sim,
            viewer,
            0.8,
        )

        # ----------------------------------------
        # Approach
        # ----------------------------------------

        print(
            "\n[3] Approaching fingertip..."
        )

        approach_start_time = (
            data.time
        )

        approach_duration = 3.0

        first_contact_pos = None

        print_counter = 0

        while viewer.is_running():

            elapsed = (
                data.time
                - approach_start_time
            )

            alpha = np.clip(
                elapsed
                / approach_duration,
                0.0,
                1.0,
            )

            # Smooth interpolation
            s = (
                alpha
                * alpha
                * (
                    3.0
                    - 2.0 * alpha
                )
            )

            cylinder_pos = (
                start_pos
                + s
                * (
                    deep_target
                    - start_pos
                )
            )

            data.mocap_pos[
                mocap_id
            ] = cylinder_pos

            realtime_step(
                sim
            )

            report = (
                read_cylinder_contacts(
                    sim,
                    cylinder_gid,
                )
            )

            print_counter += 1

            if (
                print_counter % 25 == 0
                or report["normal"] > 0.01
            ):

                print(
                    f"\r"
                    f"  t={elapsed:5.2f}s "
                    f"contacts="
                    f"{len(report['contacts']):2d} "
                    f"Fn="
                    f"{report['normal']:7.3f} N "
                    f"Ft="
                    f"{report['tangent']:7.3f} N",
                    end="",
                    flush=True,
                )

            if (
                report["normal"]
                > 0.05
            ):

                first_contact_pos = (
                    data.mocap_pos[
                        mocap_id
                    ].copy()
                )

                print(
                    "\n\n"
                    "[CONTACT DETECTED]"
                )

                print(
                    " cylinder pos:",
                    np.round(
                        first_contact_pos,
                        4,
                    ).tolist()
                )

                print(
                    f" total normal force: "
                    f"{report['normal']:.4f} N"
                )

                for (
                    finger,
                    values,
                ) in report[
                    "per_finger"
                ].items():

                    print(
                        f"   {finger:<7s}: "
                        f"contacts="
                        f"{values['count']} "
                        f"Fn="
                        f"{values['normal']:.4f} N "
                        f"Ft="
                        f"{values['tangent']:.4f} N"
                    )

                break

            if alpha >= 1.0:

                print(
                    "\n\n[WARN] "
                    "Reached final approach "
                    "position without force."
                )

                break

            if print_counter % 10 == 0:
                viewer.sync()

        # ----------------------------------------
        # Press 4 mm further after first contact.
        # ----------------------------------------

        if (
            first_contact_pos
            is not None
            and viewer.is_running()
        ):

            print(
                "\n[4] Pressing 4 mm further..."
            )

            press_start = (
                first_contact_pos.copy()
            )

            press_target = (
                press_start
                - outward * 0.004
            )

            press_t0 = (
                data.time
            )

            press_duration = 1.0

            count = 0

            while viewer.is_running():

                elapsed = (
                    data.time
                    - press_t0
                )

                alpha = np.clip(
                    elapsed
                    / press_duration,
                    0.0,
                    1.0,
                )

                s = (
                    alpha
                    * alpha
                    * (
                        3.0
                        - 2.0 * alpha
                    )
                )

                data.mocap_pos[
                    mocap_id
                ] = (
                    press_start
                    + s
                    * (
                        press_target
                        - press_start
                    )
                )

                realtime_step(
                    sim
                )

                report = (
                    read_cylinder_contacts(
                        sim,
                        cylinder_gid,
                    )
                )

                count += 1

                if count % 25 == 0:

                    print(
                        f"\r"
                        f"  press="
                        f"{alpha*4.0:4.1f} mm "
                        f"Fn="
                        f"{report['normal']:7.3f} N "
                        f"Ft="
                        f"{report['tangent']:7.3f} N",
                        end="",
                        flush=True,
                    )

                if count % 10 == 0:
                    viewer.sync()

                if alpha >= 1.0:
                    break

            print(
                "\n"
            )

            # ------------------------------------
            # Hold contact
            # ------------------------------------

            print(
                "[5] Holding contact for 2 s..."
            )

            end_time = (
                data.time
                + 2.0
            )

            count = 0

            while (
                viewer.is_running()
                and data.time < end_time
            ):

                realtime_step(
                    sim
                )

                report = (
                    read_cylinder_contacts(
                        sim,
                        cylinder_gid,
                    )
                )

                count += 1

                if count % 50 == 0:

                    print(
                        f"  Fn="
                        f"{report['normal']:.4f} N "
                        f"Ft="
                        f"{report['tangent']:.4f} N "
                        f"contacts="
                        f"{len(report['contacts'])}"
                    )

                if count % 10 == 0:
                    viewer.sync()

            # ------------------------------------
            # Retreat
            # ------------------------------------

            print(
                "\n[6] Retreating cylinder..."
            )

            retreat_start = (
                data.mocap_pos[
                    mocap_id
                ].copy()
            )

            retreat_t0 = (
                data.time
            )

            duration = 1.5

            while viewer.is_running():

                alpha = np.clip(
                    (
                        data.time
                        - retreat_t0
                    )
                    / duration,
                    0.0,
                    1.0,
                )

                data.mocap_pos[
                    mocap_id
                ] = (
                    retreat_start
                    + alpha
                    * (
                        start_pos
                        - retreat_start
                    )
                )

                realtime_step(
                    sim
                )

                if int(
                    (
                        data.time
                        - retreat_t0
                    )
                    / sim.timestep
                ) % 10 == 0:

                    viewer.sync()

                if alpha >= 1.0:
                    break

            final_report = (
                read_cylinder_contacts(
                    sim,
                    cylinder_gid,
                )
            )

            print(
                "\n[FINAL]"
            )

            print(
                f"contacts="
                f"{len(final_report['contacts'])}"
            )

            print(
                f"normal force="
                f"{final_report['normal']:.6f} N"
            )

            print(
                "\nExpected after retreat:"
            )

            print(
                "  contacts = 0"
            )

            print(
                "  normal force ~= 0 N"
            )

        # Keep viewer open briefly
        if viewer.is_running():

            hold(
                sim,
                viewer,
                1.0,
            )


if __name__ == "__main__":
    main()
