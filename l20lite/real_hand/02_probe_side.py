#!/usr/bin/env python3

from linkerbot import L20lite
from linkerbot.exceptions import TimeoutError
import time


def test_side(side):
    print()
    print("=" * 60)
    print(f"Testing side = {side}")
    print("=" * 60)

    try:
        with L20lite(
            side=side,
            interface_name="can0",
        ) as hand:

            time.sleep(0.2)

            try:
                data = hand.angle.get_blocking(timeout_ms=1500)
                print(f"[SUCCESS] side={side}")
                print("angles =", data.angles.to_list())
                return True

            except TimeoutError:
                print(f"[TIMEOUT] side={side}")
                return False

    except Exception as e:
        print(f"[ERROR] side={side}: {type(e).__name__}: {e}")
        return False


def main():
    left_ok = test_side("left")

    time.sleep(0.5)

    right_ok = test_side("right")

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print("left :", left_ok)
    print("right:", right_ok)


if __name__ == "__main__":
    main()
