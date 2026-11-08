#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np

from linkerbot import L20lite
from linkerbot.exceptions import TimeoutError


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from l20lite_sim import L20LiteMuJoCo
from l20lite_joint_mapping import (
    REAL_SDK_ORDER,
    MUJOCO_JOINT_ORDER,
    real_sdk_to_mujoco,
)



MODEL_PATH = (
    PROJECT_ROOT
    / "generated"
    / "dual_l20lite_dynamic.xml"
)

# ------------------------------------------------------------
# Hardware
# ------------------------------------------------------------

HAND_SIDE = "right"
CAN_INTERFACE = "can0"

# Real-hand read target rate.
READ_HZ = 50.0

# MuJoCo viewer/report
PRINT_HZ = 2.0


# ============================================================
# Shared real-hand state
# ============================================================

class RealHandState:

    def __init__(self):

        self.lock = threading.Lock()

        self.q_real = None

        self.last_update = 0.0

        self.read_count = 0
        self.timeout_count = 0

        self.error = None


# ============================================================
# Real hand reader
# ============================================================

def real_hand_reader(
    state: RealHandState,
    stop_event: threading.Event,
):

    period = 1.0 / READ_HZ

    try:

        # IMPORTANT:
        #
        # This context is READ-ONLY in this program.
        #
        # We NEVER call:
        #
        #     hand.angle.set_angles(...)
        #
        # Therefore this script cannot intentionally
        # command the real hand.

        with L20lite(
            side=HAND_SIDE,
            interface_name=CAN_INTERFACE,
        ) as hand:

            while not stop_event.is_set():

                loop_start = time.perf_counter()

                try:

                    data = (
                        hand.angle.get_blocking(
                            timeout_ms=100,
                        )
                    )

                    q = np.asarray(
                        data.angles.to_list(),
                        dtype=np.float64,
                    )

                    if q.shape != (10,):
                        raise RuntimeError(
                            "Real hand returned "
                            f"unexpected shape {q.shape}"
                        )

                    with state.lock:

                        state.q_real = q.copy()

                        state.last_update = (
                            time.perf_counter()
                        )

                        state.read_count += 1

                except TimeoutError:

                    with state.lock:
                        state.timeout_count += 1

                elapsed = (
                    time.perf_counter()
                    - loop_start
                )

                sleep_time = (
                    period - elapsed
                )

                if sleep_time > 0:
                    time.sleep(
                        sleep_time
                    )

    except Exception as exc:

        with state.lock:
            state.error = repr(exc)

        stop_event.set()


# ============================================================
# Snapshot helper
# ============================================================

def get_snapshot(
    state: RealHandState,
):

    with state.lock:

        q_real = (
            None
            if state.q_real is None
            else state.q_real.copy()
        )

        return (
            q_real,
            state.last_update,
            state.read_count,
            state.timeout_count,
            state.error,
        )


# ============================================================
# Pretty status
# ============================================================

def print_status(
    q_real,
    q_sim,
    read_count,
    timeout_count,
):

    print()
    print(
        "=" * 76
    )

    print(
        f"Real -> MuJoCo    "
        f"reads={read_count} "
        f"timeouts={timeout_count}"
    )

    print(
        "=" * 76
    )

    for i in range(10):

        print(
            f"[{i}] "
            f"{REAL_SDK_ORDER[i]:<14s} "
            f"{q_real[i]:7.2f}"
            f"  ->  "
            f"{MUJOCO_JOINT_ORDER[i]:<22s} "
            f"{q_sim[i]:7.2f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )

    print(
        " L20Lite RIGHT real hand -> MuJoCo realtime mirror"
    )

    print(
        "============================================================"
    )

    print()

    print(
        "Real hand:"
    )

    print(
        f"  side      = {HAND_SIDE}"
    )

    print(
        f"  interface = {CAN_INTERFACE}"
    )

    print(
        f"  read rate = {READ_HZ:.1f} Hz"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  Real hand is READ ONLY."
    )

    print(
        "  No set_angles() command is sent to hardware."
    )

    print()

    # --------------------------------------------------------
    # Start hardware reader
    # --------------------------------------------------------

    state = RealHandState()

    stop_event = threading.Event()

    reader = threading.Thread(
        target=real_hand_reader,
        args=(
            state,
            stop_event,
        ),
        daemon=True,
    )

    reader.start()

    # --------------------------------------------------------
    # Wait for first valid real-hand frame
    # --------------------------------------------------------

    print(
        "[REAL] Waiting for first 10-D frame..."
    )

    while not stop_event.is_set():

        (
            q_real,
            _,
            _,
            _,
            error,
        ) = get_snapshot(
            state
        )

        if error is not None:

            raise RuntimeError(
                f"Real-hand reader failed: {error}"
            )

        if q_real is not None:
            break

        time.sleep(
            0.02
        )

    if q_real is None:

        raise RuntimeError(
            "No valid real-hand data."
        )

    print(
        "[REAL] First frame received:"
    )

    print(
        np.round(
            q_real,
            2,
        ).tolist()
    )

    # --------------------------------------------------------
    # Map initial real posture
    # --------------------------------------------------------

    q_sim = real_sdk_to_mujoco(
        q_real
    )

    print()

    print(
        "[MAP] Initial MuJoCo command:"
    )

    print(
        np.round(
            q_sim,
            2,
        ).tolist()
    )

    # --------------------------------------------------------
    # MuJoCo
    # --------------------------------------------------------

    sim = L20LiteMuJoCo(
        MODEL_PATH
    )

    zero = np.zeros(
        10,
        dtype=np.float64,
    )

    sim.left.angle.set_angles(
        zero
    )

    sim.right.angle.set_angles(
        q_sim
    )

    # Settle to initial measured real posture.
    for _ in range(500):

        sim.left.angle.set_angles(
            zero
        )

        sim.right.angle.set_angles(
            q_sim
        )

        sim.step()

    print()

    print(
        "[SIM] MuJoCo ready."
    )

    print()

    print(
        "Move the REAL RIGHT hand manually."
    )

    print(
        "MuJoCo RIGHT hand should follow it."
    )

    print()

    print(
        "Close the viewer or press Ctrl+C to stop."
    )

    # --------------------------------------------------------
    # Viewer
    # --------------------------------------------------------

    try:

        with mujoco.viewer.launch_passive(
            sim.model,
            sim.data,
        ) as viewer:

            # Focus on right hand.
            viewer.cam.lookat[:] = (
                np.array(
                    [
                        0.0,
                        -0.18,
                        0.105,
                    ],
                    dtype=np.float64,
                )
            )

            viewer.cam.distance = 0.30
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -18

            viewer.sync()

            step_count = 0

            next_print = (
                time.perf_counter()
            )

            while (
                viewer.is_running()
                and not stop_event.is_set()
            ):

                wall_start = (
                    time.perf_counter()
                )

                (
                    q_real,
                    last_update,
                    read_count,
                    timeout_count,
                    error,
                ) = get_snapshot(
                    state
                )

                if error is not None:

                    print(
                        "\n[ERROR] Real hand reader:"
                    )

                    print(
                        error
                    )

                    break

                # --------------------------------------------
                # Real SDK -> MuJoCo
                # --------------------------------------------

                if q_real is not None:

                    q_sim = (
                        real_sdk_to_mujoco(
                            q_real
                        )
                    )

                    sim.right.angle.set_angles(
                        q_sim
                    )

                # Left hand remains neutral.
                sim.left.angle.set_angles(
                    zero
                )

                # --------------------------------------------
                # Physics
                # --------------------------------------------

                sim.step()

                step_count += 1

                if step_count % 10 == 0:

                    viewer.sync()

                # --------------------------------------------
                # Status ~2 Hz
                # --------------------------------------------

                now = time.perf_counter()

                if (
                    q_real is not None
                    and now >= next_print
                ):

                    print_status(
                        q_real,
                        q_sim,
                        read_count,
                        timeout_count,
                    )

                    next_print = (
                        now
                        + 1.0 / PRINT_HZ
                    )

                # --------------------------------------------
                # Real-time simulation pacing
                # --------------------------------------------

                elapsed = (
                    time.perf_counter()
                    - wall_start
                )

                sleep_time = (
                    sim.timestep
                    - elapsed
                )

                if sleep_time > 0:

                    time.sleep(
                        sleep_time
                    )

    except KeyboardInterrupt:

        print(
            "\n[STOP] Ctrl+C"
        )

    finally:

        stop_event.set()

        reader.join(
            timeout=1.0
        )

        print()
        print(
            "Realtime mirror stopped."
        )

        print(
            "No command was sent to the real hand."
        )


if __name__ == "__main__":
    main()
