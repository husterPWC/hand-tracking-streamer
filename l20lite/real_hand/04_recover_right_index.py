#!/usr/bin/env python3

import time
from linkerbot import L20lite
from linkerbot.exceptions import TimeoutError

INDEX_FLEX_ID = 2

with L20lite(side="right", interface_name="can0") as hand:

    try:
        data = hand.angle.get_blocking(timeout_ms=1000)
    except TimeoutError:
        print("[FAIL] Cannot read hand state. No command sent.")
        raise SystemExit(1)

    q = data.angles.to_list()

    current = float(q[INDEX_FLEX_ID])
    target = 100.0

    print("Current 10-D:", q)
    print(f"Recover index_flex: {current:.2f} -> {target:.2f}")
    print("Only index_flex will change.")

    value = current

    while value < target:
        value = min(value + 1.0, target)

        q[INDEX_FLEX_ID] = value
        hand.angle.set_angles(q)

        print(f"index_flex = {value:.2f}")
        time.sleep(0.25)

    print("[DONE] index_flex returned to 100.")
