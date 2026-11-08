#!/usr/bin/env python3

import time
from linkerbot import L20lite
from linkerbot.exceptions import TimeoutError


TESTS = [
    # id, name, delta
    (0, "thumb_flex", -20.0),
    (1, "thumb_abd",  +15.0),
    (9, "thumb_yaw",  +15.0),
]

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
    print("L20Lite RIGHT - thumb channel identification")
    print("Test: thumb_flex / thumb_abd / thumb_yaw")
    print("=" * 65)

    with L20lite(side="right", interface_name="can0") as hand:

        for joint_id, name, delta in TESTS:

            try:
                data = hand.angle.get_blocking(timeout_ms=1000)
            except TimeoutError:
                print("[FAIL] Cannot read hand angles.")
                return

            q0 = data.angles.to_list()

            current = float(q0[joint_id])
            target = max(0.0, min(100.0, current + delta))

            print()
            print("-" * 65)
            print(f"Testing [{joint_id}] {name}")
            print(f"{current:.2f} -> {target:.2f}")
            print("-" * 65)

            input("Press ENTER to move...")

            move_channel(hand, q0, joint_id, target)

            print()
            print(f"{name} is now at about {target:.2f}")
            print("Observe the exact thumb motion.")

            input("\nPress ENTER to restore...")

            try:
                data_now = hand.angle.get_blocking(timeout_ms=1000)
                q_now = data_now.angles.to_list()
            except TimeoutError:
                q_now = list(q0)
                q_now[joint_id] = target

            move_channel(hand, q_now, joint_id, current)

            print(f"[RESTORED] {name} -> {current:.2f}")

    print()
    print("=" * 65)
    print("Thumb tests completed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
