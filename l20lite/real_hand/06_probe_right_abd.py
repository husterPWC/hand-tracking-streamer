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

TEST_CHANNELS = [
    (6, "index_abd"),
    (7, "ring_abd"),
    (8, "pinky_abd"),
]

DELTA = +20.0
STEP = 2.0
STEP_TIME = 0.15


def move_channel(hand, q_base, joint_id, target):
    q = list(q_base)
    current = float(q[joint_id])

    if target > current:
        value = current
        while value < target:
            value = min(value + STEP, target)
            q[joint_id] = value
            hand.angle.set_angles(q)
            time.sleep(STEP_TIME)
    else:
        value = current
        while value > target:
            value = max(value - STEP, target)
            q[joint_id] = value
            hand.angle.set_angles(q)
            time.sleep(STEP_TIME)


def main():
    print("=" * 65)
    print("L20Lite RIGHT - ABD channel identification")
    print("Test: index_abd / ring_abd / pinky_abd")
    print("=" * 65)

    with L20lite(
        side="right",
        interface_name="can0",
    ) as hand:

        for joint_id, name in TEST_CHANNELS:

            try:
                data = hand.angle.get_blocking(timeout_ms=1000)
            except TimeoutError:
                print("[FAIL] Cannot read hand angles.")
                return

            q0 = data.angles.to_list()

            current = float(q0[joint_id])
            target = min(100.0, current + DELTA)

            print()
            print("-" * 65)
            print(f"Testing [{joint_id}] {name}")
            print(f"{current:.2f} -> {target:.2f}")
            print("-" * 65)

            input("Press ENTER to move...")

            move_channel(
                hand,
                q0,
                joint_id,
                target,
            )

            print()
            print(f"{name} is now at about {target:.2f}")
            print()
            print("Observe:")
            print("  1. Which finger moved?")
            print("  2. Which lateral direction did it move?")
            print("  3. Did the finger move toward or away from the middle finger?")

            input("\nPress ENTER to restore...")

            try:
                data_now = hand.angle.get_blocking(timeout_ms=1000)
                q_now = data_now.angles.to_list()
            except TimeoutError:
                q_now = list(q0)
                q_now[joint_id] = target

            move_channel(
                hand,
                q_now,
                joint_id,
                current,
            )

            print(f"[RESTORED] {name} -> {current:.2f}")

        print()
        print("=" * 65)
        print("ABD tests completed.")
        print("=" * 65)


if __name__ == "__main__":
    main()
