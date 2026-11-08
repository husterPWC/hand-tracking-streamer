#!/usr/bin/env python3

from linkerbot import L20lite
from linkerbot.exceptions import TimeoutError


JOINT_NAMES = [
    "thumb_flex",     # 0 拇指弯曲
    "thumb_abd",      # 1 拇指侧摆
    "index_flex",     # 2 食指弯曲
    "middle_flex",    # 3 中指弯曲
    "ring_flex",      # 4 无名指弯曲
    "pinky_flex",     # 5 小指弯曲
    "index_abd",      # 6 食指侧摆
    "ring_abd",       # 7 无名指侧摆
    "pinky_abd",      # 8 小指侧摆
    "thumb_yaw",      # 9 拇指旋转
]


def main():
    print("=" * 60)
    print("Linker L20Lite RIGHT hand - READ ONLY test")
    print("CAN interface : can0")
    print("Side          : right")
    print("NO motion command will be sent.")
    print("=" * 60)

    try:
        with L20lite(
            side="right",
            interface_name="can0",
        ) as hand:

            print("\n[OK] L20Lite object created.")

            print("\nReading joint angles...")

            data = hand.angle.get_blocking(timeout_ms=1000)
            angles = data.angles.to_list()

            print("\nRaw 10-D angles:")
            print(angles)

            print("\nJoint mapping:")
            for i, (name, value) in enumerate(zip(JOINT_NAMES, angles)):
                print(f"  [{i:2d}] {name:14s} = {value}")

            print("\n[PASS] Left L20Lite communication is working.")

    except TimeoutError:
        print("\n[FAIL] Timeout while reading L20Lite.")
        print("Check:")
        print("  1. Hand power")
        print("  2. USB-CAN connection")
        print("  3. can0 is UP")
        print("  4. bitrate = 1000000")
        print("  5. side='left'")

    except Exception as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
