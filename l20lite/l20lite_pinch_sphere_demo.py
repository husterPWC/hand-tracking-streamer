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
    / "dual_l20lite_pinch_sphere.xml"
)


# ============================================================
# Natural pinch trajectory
#
# Use the current manually corrected thumb semantics:
#
# roll:
#     opposition toward index
#
# yaw:
#     toward palm
#
# pitch:
#     final closure + fingertip flex
# ============================================================

HAND_OPEN = np.array(
    [
        0,
        0,

        4,

        6,
        8,
        10,

        0,
        0,
        0,

        0,
    ],
    dtype=np.float64,
)


THUMB_OPPOSITION = np.array(
    [
        3,

        18,

        7,

        8,
        10,
        13,

        0,
        0,
        0,

        24,
    ],
    dtype=np.float64,
)


PINCH_PRESHAPE = np.array(
    [
        8,

        30,

        13,

        11,
        14,
        17,

        0,
        0,
        0,

        32,
    ],
    dtype=np.float64,
)


PINCH_CLOSE = np.array(
    [
        45,

        36,

        50,

        14,
        17,
        20,

        0,
        0,
        0,

        38,
    ],
    dtype=np.float64,
)


CONTACT_THRESHOLD = 0.05

SPHERE_RADIUS = 0.012

SPHERE_MASS = 0.030


# ============================================================
# Helpers
# ============================================================

def smoothstep(x):

    x = float(
        np.clip(
            x,
            0.0,
            1.0,
        )
    )

    return (
        x
        * x
        * (
            3.0
            - 2.0 * x
        )
    )


def interpolate(
    q0,
    q1,
    alpha,
):

    s = smoothstep(
        alpha
    )

    return (
        q0
        + s
        * (
            q1
            - q0
        )
    )


def realtime_step(
    sim,
):

    t0 = time.perf_counter()

    sim.step()

    elapsed = (
        time.perf_counter()
        - t0
    )

    wait = (
        sim.timestep
        - elapsed
    )

    if wait > 0:

        time.sleep(
            wait
        )


def fast_step(
    sim,
    steps,
):

    for _ in range(
        steps
    ):

        sim.step()


def hold(
    sim,
    viewer,
    duration,
):

    end_time = (
        sim.data.time
        + duration
    )

    counter = 0

    while (
        viewer.is_running()
        and sim.data.time < end_time
    ):

        realtime_step(
            sim
        )

        counter += 1

        if counter % 10 == 0:

            viewer.sync()


def site_id(
    model,
    name,
):

    sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name,
    )

    if sid < 0:

        raise KeyError(
            f"Missing site: {name}"
        )

    return sid


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
            f"Missing geom: {name}"
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
            f"Missing body: {name}"
        )

    return bid


def joint_id(
    model,
    name,
):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:

        raise KeyError(
            f"Missing joint: {name}"
        )

    return jid


def equality_id(
    model,
    name,
):

    eid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        name,
    )

    if eid < 0:

        raise KeyError(
            f"Missing equality: {name}"
        )

    return eid


# ============================================================
# Contacts
# ============================================================

def classify_finger(
    geom_name,
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

    return "other"


def read_sphere_contacts(
    sim,
    sphere_gid,
):

    model = sim.model
    data = sim.data

    result = {}

    for cid in range(
        data.ncon
    ):

        c = data.contact[
            cid
        ]

        g1 = int(
            c.geom1
        )

        g2 = int(
            c.geom2
        )

        if (
            g1 != sphere_gid
            and g2 != sphere_gid
        ):

            continue

        other_gid = (
            g2
            if g1 == sphere_gid
            else g1
        )

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            other_gid,
        )

        if not name:

            continue

        if not name.startswith(
            "right_"
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

        fn = abs(
            float(
                force6[0]
            )
        )

        ft = float(
            np.linalg.norm(
                force6[1:3]
            )
        )

        finger = classify_finger(
            name
        )

        if finger not in result:

            result[
                finger
            ] = {
                "Fn": 0.0,
                "Ft": 0.0,
                "count": 0,
            }

        result[
            finger
        ]["Fn"] += fn

        result[
            finger
        ]["Ft"] += ft

        result[
            finger
        ]["count"] += 1

    return result


def finger_force(
    report,
    finger,
):

    return (
        report
        .get(
            finger,
            {}
        )
        .get(
            "Fn",
            0.0,
        )
    )


# ============================================================
# Move helper
# ============================================================

def move_hand(
    sim,
    viewer,
    q0,
    q1,
    duration,
    label,
):

    print(
        f"\n[MOVE] {label}"
    )

    t0 = (
        sim.data.time
    )

    counter = 0

    while viewer.is_running():

        alpha = np.clip(
            (
                sim.data.time
                - t0
            )
            / duration,
            0.0,
            1.0,
        )

        cmd = interpolate(
            q0,
            q1,
            alpha,
        )

        sim.right.angle.set_angles(
            cmd
        )

        realtime_step(
            sim
        )

        counter += 1

        if counter % 10 == 0:

            viewer.sync()

        if alpha >= 1.0:

            break


# ============================================================
# Find pinch center using REAL TIP SITES
# ============================================================

def find_pinch_center(
    sim,
    thumb_sid,
    index_sid,
    object_qadr,
    object_dadr,
    weld_id,
):

    print(
        "\n========================================"
    )

    print(
        "[SEARCH] Pinch center from tip sites"
    )

    print(
        "========================================"
    )

    sim.reset()

    sim.model.opt.gravity[:] = (
        0.0,
        0.0,
        0.0,
    )

    sim.data.eq_active[
        weld_id
    ] = 0

    # Move sphere far away.
    sim.data.qpos[
        object_qadr:
        object_qadr + 7
    ] = np.array(
        [
            1,
            1,
            1,

            1,
            0,
            0,
            0,
        ],
        dtype=np.float64,
    )

    sim.data.qvel[
        object_dadr:
        object_dadr + 6
    ] = 0.0

    sim.left.angle.set_angles(
        HAND_OPEN
    )

    sim.right.angle.set_angles(
        PINCH_PRESHAPE
    )

    fast_step(
        sim,
        300,
    )

    samples = []

    for alpha in np.linspace(
        0.0,
        1.0,
        61,
    ):

        cmd = interpolate(
            PINCH_PRESHAPE,
            PINCH_CLOSE,
            alpha,
        )

        sim.right.angle.set_angles(
            cmd
        )

        fast_step(
            sim,
            80,
        )

        mujoco.mj_forward(
            sim.model,
            sim.data,
        )

        thumb = np.asarray(
            sim.data.site_xpos[
                thumb_sid
            ],
            dtype=np.float64,
        ).copy()

        index = np.asarray(
            sim.data.site_xpos[
                index_sid
            ],
            dtype=np.float64,
        ).copy()

        gap = float(
            np.linalg.norm(
                index
                - thumb
            )
        )

        center = (
            0.5
            * (
                thumb
                + index
            )
        )

        samples.append(
            {
                "alpha":
                    float(alpha),

                "gap":
                    gap,

                "thumb":
                    thumb,

                "index":
                    index,

                "center":
                    center,

                "cmd":
                    cmd.copy(),
            }
        )

    gaps = np.asarray(
        [
            x["gap"]
            for x in samples
        ]
    )

    minimum_index = int(
        np.argmin(
            gaps
        )
    )

    minimum = (
        samples[
            minimum_index
        ]
    )

    # Sphere diameter 24 mm.
    #
    # Use a reference slightly before full contact.
    desired_gap = (
        2.0 * SPHERE_RADIUS
        + 0.004
    )

    candidate_index = None

    for i, sample in enumerate(
        samples
    ):

        if sample[
            "gap"
        ] <= desired_gap:

            candidate_index = i
            break

    if candidate_index is None:

        candidate_index = (
            minimum_index
        )

        print(
            "\n[WARN]"
        )

        print(
            "Tip trajectory does not reach "
            "the desired 28 mm gap."
        )

    candidate = (
        samples[
            candidate_index
        ]
    )

    print(
        f"preshape gap = "
        f"{samples[0]['gap']*1000:.1f} mm"
    )

    print(
        f"minimum gap  = "
        f"{minimum['gap']*1000:.1f} mm"
    )

    print(
        f"target gap   = "
        f"{desired_gap*1000:.1f} mm"
    )

    print(
        f"selected gap = "
        f"{candidate['gap']*1000:.1f} mm"
    )

    print(
        f"alpha        = "
        f"{candidate['alpha']:.3f}"
    )

    print(
        "thumb tip:",
        np.round(
            candidate[
                "thumb"
            ],
            4,
        ).tolist()
    )

    print(
        "index tip:",
        np.round(
            candidate[
                "index"
            ],
            4,
        ).tolist()
    )

    print(
        "sphere center:",
        np.round(
            candidate[
                "center"
            ],
            4,
        ).tolist()
    )

    return candidate


# ============================================================
# Main
# ============================================================

def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            MODEL_PATH
        )

    sim = L20LiteMuJoCo(
        MODEL_PATH
    )

    model = sim.model
    data = sim.data

    thumb_sid = site_id(
        model,
        "right_thumb_tip_site",
    )

    index_sid = site_id(
        model,
        "right_index_tip_site",
    )

    sphere_gid = geom_id(
        model,
        "pinch_sphere_geom",
    )

    sphere_jid = joint_id(
        model,
        "pinch_sphere_free",
    )

    object_qadr = int(
        model.jnt_qposadr[
            sphere_jid
        ]
    )

    object_dadr = int(
        model.jnt_dofadr[
            sphere_jid
        ]
    )

    mocap_bid = body_id(
        model,
        "pinch_target_mocap",
    )

    mocap_id = int(
        model.body_mocapid[
            mocap_bid
        ]
    )

    weld_id = equality_id(
        model,
        "pinch_sphere_weld",
    )

    print(
        "========================================"
    )

    print(
        " L20 Lite Natural Pinch - Free Sphere"
    )

    print(
        "========================================"
    )

    print(
        f"object mass   = "
        f"{SPHERE_MASS*1000:.0f} g"
    )

    print(
        f"object weight = "
        f"{SPHERE_MASS*9.81:.3f} N"
    )

    # ========================================================
    # 1. Determine location from the final natural pinch path
    # ========================================================

    candidate = find_pinch_center(
        sim,
        thumb_sid,
        index_sid,
        object_qadr,
        object_dadr,
        weld_id,
    )

    center = candidate[
        "center"
    ].copy()

    # ========================================================
    # 2. Reset visible experiment
    # ========================================================

    sim.reset()

    model.opt.gravity[:] = (
        0.0,
        0.0,
        0.0,
    )

    data.eq_active[
        weld_id
    ] = 1

    # Sphere orientation irrelevant.
    data.qpos[
        object_qadr:
        object_qadr + 7
    ] = np.array(
        [
            center[0],
            center[1],
            center[2],

            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    data.qvel[
        object_dadr:
        object_dadr + 6
    ] = 0.0

    data.mocap_pos[
        mocap_id
    ] = center

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

    sim.left.angle.set_angles(
        HAND_OPEN
    )

    sim.right.angle.set_angles(
        HAND_OPEN
    )

    mujoco.mj_forward(
        model,
        data,
    )

    # ========================================================
    # Viewer
    # ========================================================

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # Turn sites off now so view is clean.
        viewer.opt.sitegroup[:] = 0

        viewer.cam.lookat[:] = (
            center
        )

        viewer.cam.distance = 0.27
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -8

        viewer.sync()

        # ----------------------------------------------------
        # Natural sequence
        # ----------------------------------------------------

        print(
            "\n[1] Natural open..."
        )

        hold(
            sim,
            viewer,
            0.5,
        )

        move_hand(
            sim,
            viewer,
            HAND_OPEN,
            THUMB_OPPOSITION,
            1.0,
            "THUMB OPPOSITION",
        )

        move_hand(
            sim,
            viewer,
            THUMB_OPPOSITION,
            PINCH_PRESHAPE,
            1.0,
            "PINCH PRESHAPE",
        )

        hold(
            sim,
            viewer,
            0.4,
        )

        # ----------------------------------------------------
        # Contact-driven natural closure
        # ----------------------------------------------------

        print(
            "\n[2] Contact-driven coordinated closure..."
        )

        start_time = (
            data.time
        )

        duration = 4.0

        bilateral_count = 0

        detected_cmd = None

        counter = 0

        while viewer.is_running():

            alpha = np.clip(
                (
                    data.time
                    - start_time
                )
                / duration,
                0.0,
                1.0,
            )

            cmd = interpolate(
                PINCH_PRESHAPE,
                PINCH_CLOSE,
                alpha,
            )

            sim.right.angle.set_angles(
                cmd
            )

            realtime_step(
                sim
            )

            report = read_sphere_contacts(
                sim,
                sphere_gid,
            )

            thumb_force = finger_force(
                report,
                "thumb",
            )

            index_force = finger_force(
                report,
                "index",
            )

            if (
                thumb_force > CONTACT_THRESHOLD
                and
                index_force > CONTACT_THRESHOLD
            ):

                bilateral_count += 1

            else:

                bilateral_count = 0

            counter += 1

            if counter % 50 == 0:

                print(
                    f"  close="
                    f"{alpha*100:5.1f}% "
                    f"thumb="
                    f"{thumb_force:6.3f} N "
                    f"index="
                    f"{index_force:6.3f} N"
                )

            if bilateral_count >= 20:

                detected_cmd = (
                    cmd.copy()
                )

                print(
                    "\n[BILATERAL CONTACT]"
                )

                print(
                    f"alpha    = "
                    f"{alpha:.3f}"
                )

                print(
                    f"thumb Fn = "
                    f"{thumb_force:.4f} N"
                )

                print(
                    f"index Fn = "
                    f"{index_force:.4f} N"
                )

                print(
                    "command =",
                    np.round(
                        detected_cmd,
                        1,
                    ).tolist()
                )

                break

            # Do not crush a welded object.
            if (
                max(
                    thumb_force,
                    index_force,
                )
                > 5.0
                and
                min(
                    thumb_force,
                    index_force,
                )
                < 0.02
            ):

                print(
                    "\n[ONE-SIDED FORCE LIMIT]"
                )

                print(
                    f"thumb="
                    f"{thumb_force:.3f} N "
                    f"index="
                    f"{index_force:.3f} N"
                )

                break

            if alpha >= 1.0:

                break

            if counter % 10 == 0:

                viewer.sync()

        if detected_cmd is None:

            print(
                "\n========================================"
            )

            print(
                "PINCH RESULT: NO BILATERAL CONTACT"
            )

            print(
                "========================================"
            )

            hold(
                sim,
                viewer,
                2.0,
            )

            return

        # ----------------------------------------------------
        # Small coordinated extra squeeze
        # ----------------------------------------------------

        squeeze_cmd = (
            detected_cmd
            + 0.04
            * (
                PINCH_CLOSE
                - detected_cmd
            )
        )

        squeeze_cmd = np.clip(
            squeeze_cmd,
            0.0,
            100.0,
        )

        move_hand(
            sim,
            viewer,
            detected_cmd,
            squeeze_cmd,
            0.3,
            "SMALL SQUEEZE",
        )

        hold(
            sim,
            viewer,
            0.25,
        )

        report = read_sphere_contacts(
            sim,
            sphere_gid,
        )

        print(
            "\n[3] Before release"
        )

        print(
            f"thumb="
            f"{finger_force(report, 'thumb'):.4f} N"
        )

        print(
            f"index="
            f"{finger_force(report, 'index'):.4f} N"
        )

        # ----------------------------------------------------
        # Release sphere
        # ----------------------------------------------------

        before = (
            data.qpos[
                object_qadr:
                object_qadr + 3
            ].copy()
        )

        data.eq_active[
            weld_id
        ] = 0

        model.opt.gravity[:] = (
            0.0,
            0.0,
            -9.81,
        )

        sim.right.angle.set_angles(
            squeeze_cmd
        )

        print(
            "\n[4] Weld OFF, gravity ON."
        )

        print(
            "Sphere is now completely free."
        )

        # ----------------------------------------------------
        # Hold 2 sec
        # ----------------------------------------------------

        end_time = (
            data.time
            + 2.0
        )

        counter = 0

        while (
            viewer.is_running()
            and data.time < end_time
        ):

            sim.right.angle.set_angles(
                squeeze_cmd
            )

            realtime_step(
                sim
            )

            counter += 1

            if counter % 50 == 0:

                pos = (
                    data.qpos[
                        object_qadr:
                        object_qadr + 3
                    ]
                )

                report = read_sphere_contacts(
                    sim,
                    sphere_gid,
                )

                print(
                    f"  z={pos[2]:.4f} "
                    f"thumb="
                    f"{finger_force(report, 'thumb'):.3f} "
                    f"index="
                    f"{finger_force(report, 'index'):.3f}"
                )

            if counter % 10 == 0:

                viewer.sync()

        after = (
            data.qpos[
                object_qadr:
                object_qadr + 3
            ].copy()
        )

        report = read_sphere_contacts(
            sim,
            sphere_gid,
        )

        drop = float(
            before[2]
            - after[2]
        )

        displacement = float(
            np.linalg.norm(
                after
                - before
            )
        )

        thumb_force = finger_force(
            report,
            "thumb",
        )

        index_force = finger_force(
            report,
            "index",
        )

        print(
            "\n========================================"
        )

        print(
            "[RESULT]"
        )

        print(
            "before:",
            np.round(
                before,
                4,
            ).tolist()
        )

        print(
            "after :",
            np.round(
                after,
                4,
            ).tolist()
        )

        print(
            f"vertical drop = "
            f"{drop*1000:.2f} mm"
        )

        print(
            f"3D movement   = "
            f"{displacement*1000:.2f} mm"
        )

        print(
            f"thumb force   = "
            f"{thumb_force:.4f} N"
        )

        print(
            f"index force   = "
            f"{index_force:.4f} N"
        )

        success = (
            drop < 0.020
            and thumb_force > 0.05
            and index_force > 0.05
        )

        if success:

            print(
                "\nPINCH RESULT: SUCCESS"
            )

        else:

            print(
                "\nPINCH RESULT: NOT STABLE"
            )

        print(
            "========================================"
        )

        hold(
            sim,
            viewer,
            2.0,
        )


if __name__ == "__main__":
    main()
