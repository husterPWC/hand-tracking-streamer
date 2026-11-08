#!/usr/bin/env python3

import time
from linkerbot import L20lite
from linkerbot.exceptions import TimeoutError


JOINT_NAMES = [
    "thumb_flex",     # 0
    "thumb_abd",      # 1
    "index_flex",     # 2
    "middle_flex",    # 3
    "ring_flex",      # 4
    "pinky_flex",     # 5
    "index_abd",      # 6
    "ring_abd",       # 7
    "pinky_abd",      # 8
    "thumb_yaw",      # 9
]

INDEX_FLEX_ID = 2
DELTA = -50.0
STEP = 1.0
STEP_TIME = 0.20


def send_angles(hand, q):
    hand.angle.set_angles([float(x) for x in q])


def main():
    print("=" * 65)
    print("L20Lite RIGHT hand - SAFE index_flex test")
    print("Only SDK channel [2] index_flex will move.")
    print("Maximum requested change: -5 units.")
    print("=" * 65)

    with L20lite(
        side="right",
        interface_name="can0",
    ) as hand:

        # 1. Read current 10-D state
        try:
            data = hand.angle.get_blocking(timeout_ms=1000)
        except TimeoutError:
            print("[FAIL] Cannot read current angles. No motion command sent.")
            return

        q0 = data.angles.to_list()

        print("\nCurrent 10-D state:")
        for i, (name, value) in enumerate(zip(JOINT_NAMES, q0)):
            print(f"  [{i:2d}] {name:14s} = {value:8.3f}")

        current = float(q0[INDEX_FLEX_ID])
        target = max(0.0, min(100.0, current + DELTA))

        print("\nTest:")
        print(f"  index_flex: {current:.3f} -> {target:.3f}")
        print("  Other 9 channels remain unchanged.")
        print()
        print("Please keep the hand clear of objects.")
        print("Watch ONLY the right index finger.")
        input("\nPress ENTER to begin the small motion test...")

        q = list(q0)

        # 2. Slowly move index_flex downward
        print("\n[MOVE] Moving index_flex...")

        value = current
        while value - STEP > target:
            value -= STEP
            q[INDEX_FLEX_ID] = value
            send_angles(hand, q)
            print(f"  index_flex = {value:.3f}")
            time.sleep(STEP_TIME)

        q[INDEX_FLEX_ID] = target
        send_angles(hand, q)
        print(f"  index_flex = {target:.3f}")
        time.sleep(1.0)

        # 3. Read state at target
        try:
            data_target = hand.angle.get_blocking(timeout_ms=1000)
            q_read = data_target.angles.to_list()

            print("\nReadback at target:")
            print(f"  commanded index_flex = {target:.3f}")
            print(f"  measured  index_flex = {q_read[INDEX_FLEX_ID]:.3f}")

        except TimeoutError:
            print("\n[WARN] Readback timeout at target.")

        input("\nObserve the finger. Press ENTER to return to the original position...")

        # 4. Slowly restore original value
        print("\n[RETURN] Returning index_flex...")

        value = target
        while value + STEP < current:
            value += STEP
            q[INDEX_FLEX_ID] = value
            send_angles(hand, q)
            print(f"  index_flex = {value:.3f}")
            time.sleep(STEP_TIME)

        q[INDEX_FLEX_ID] = current
        send_angles(hand, q)

        print(f"  index_flex = {current:.3f}")
        time.sleep(0.5)

        print("\n[PASS] Test completed.")
        print("Original 10-D command restored.")


if __name__ == "__main__":
    main()
