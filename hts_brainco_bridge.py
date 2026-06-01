import argparse
import asyncio
import os
import sys
import time
import json
import numpy as np

from hand_tracking_sdk import HTSClient, HTSClientConfig, StreamOutput, TransportMode

from hts_inspect_stream import (
    compute_brainco_0_100,
    CommandSmoother,
    get_side_value,
    is_target_hand_frame,
)


# ============================================================
# Strong BrainCo / Revo SDK path
# ============================================================

DEFAULT_REVO1_DIR = os.path.expanduser(
    "~/codePWC/stark-serialport-example/python/revo1"
)

if DEFAULT_REVO1_DIR not in sys.path:
    sys.path.append(DEFAULT_REVO1_DIR)

try:
    from revo1_utils import open_modbus_revo1, libstark
except Exception:
    open_modbus_revo1 = None
    libstark = None


# ============================================================
# Channel definition
# ============================================================
#
# HTS internal command:
#   0   = open
#   100 = closed
#
# Official BrainCo SDK command:
#   0    = open
#   1000 = closed
#
# Official SDK order:
#   [Thumb, ThumbAux, Index, Middle, Ring, Pinky]
#
# Our semantic command:
#   thumb_horizontal = thumb moves toward palm
#   thumb_vertical   = thumb moves toward index
#   index
#   middle
#   ring
#   little
#
# Default mapping assumption:
#   Thumb    <- thumb_vertical
#   ThumbAux <- thumb_horizontal
#
# If the two thumb joints are swapped on the real hand, change
# THUMB_FIRST_CHANNEL and THUMB_AUX_CHANNEL below.

CHANNELS = [
    "thumb_horizontal",
    "thumb_vertical",
    "index",
    "middle",
    "ring",
    "little",
]

THUMB_FIRST_CHANNEL = "thumb_vertical"
THUMB_AUX_CHANNEL = "thumb_horizontal"


def clamp_0_100(x):
    return int(round(max(0.0, min(100.0, float(x)))))


def clamp_0_1000(x):
    return int(round(max(0.0, min(1000.0, float(x)))))


def cmd_100_to_cmd_1000(cmd_100):
    """
    Convert normalized HTS command to official BrainCo SDK command.

    Input:
        0~100, 0=open, 100=closed

    Output:
        0~1000, 0=open, 1000=closed
    """
    return {
        k: clamp_0_1000(float(cmd_100.get(k, 0)) * 10.0)
        for k in CHANNELS
    }


def cmd_to_sdk_positions(cmd_1000):
    """
    Convert semantic command dict to official SDK position list.

    Official order:
        [Thumb, ThumbAux, Index, Middle, Ring, Pinky]
    """
    return [
        clamp_0_1000(cmd_1000[THUMB_FIRST_CHANNEL]),
        clamp_0_1000(cmd_1000[THUMB_AUX_CHANNEL]),
        clamp_0_1000(cmd_1000["index"]),
        clamp_0_1000(cmd_1000["middle"]),
        clamp_0_1000(cmd_1000["ring"]),
        clamp_0_1000(cmd_1000["little"]),
    ]


def norm(v):
    return float(np.linalg.norm(v))


def np_point(p):
    return np.array(p, dtype=np.float32)


def thumb_horizontal_features(points):
    wrist = np_point(points[0])
    thumb_tip = np_point(points[4])

    index_base = np_point(points[5])
    middle_base = np_point(points[9])
    ring_base = np_point(points[13])
    little_base = np_point(points[17])

    palm_center = (wrist + index_base + middle_base + ring_base + little_base) / 5.0

    palm_width = norm(index_base - little_base)
    if palm_width < 1e-6:
        palm_width = 0.08

    across_axis = index_base - little_base
    across_axis = across_axis / max(norm(across_axis), 1e-6)

    forward_axis = middle_base - wrist
    forward_axis = forward_axis / max(norm(forward_axis), 1e-6)

    palm_normal = np.cross(across_axis, forward_axis)
    palm_normal = palm_normal / max(norm(palm_normal), 1e-6)

    rel = thumb_tip - palm_center

    return {
        "dist_thumb_palm": norm(thumb_tip - palm_center) / palm_width,
        "dist_thumb_index": norm(thumb_tip - index_base) / palm_width,
        "dist_thumb_middle": norm(thumb_tip - middle_base) / palm_width,
        "proj_across": float(np.dot(rel, across_axis)) / palm_width,
        "proj_normal": float(np.dot(rel, palm_normal)) / palm_width,
    }


def map_raw_to_100(x, x_open, x_closed):
    denom = x_closed - x_open
    if abs(denom) < 1e-6:
        return 0

    value = (x - x_open) / denom * 100.0
    return clamp_0_100(value)


def load_thumb_calibration(path):
    if not path:
        return None

    with open(path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    print("[ThumbCalib] Loaded:", path)
    print("[ThumbCalib] feature:", calib["feature"])
    print("[ThumbCalib] open:", calib["open"])
    print("[ThumbCalib] closed:", calib["closed"])

    return calib


def compute_thumb_horizontal_calibrated(points, calib):
    features = thumb_horizontal_features(points)
    feature_name = calib["feature"]

    if feature_name not in features:
        raise RuntimeError(f"Calibration feature not found: {feature_name}")

    raw = features[feature_name]
    value = map_raw_to_100(raw, calib["open"], calib["closed"])

    return value

class StepLimiter100:
    """
    Step limiter in normalized 0~100 space.

    Example:
        max_step=3 means each command changes by at most 3 per send,
        which equals 30 units in the official 0~1000 SDK space.
    """

    def __init__(self, max_step=3.0):
        self.max_step = float(max_step)
        self.last = None

    def update(self, target):
        if self.last is None:
            self.last = {k: float(target[k]) for k in CHANNELS}
            return {k: clamp_0_100(self.last[k]) for k in CHANNELS}

        out = {}

        for k in CHANNELS:
            old = float(self.last[k])
            new = float(target[k])
            delta = new - old

            if delta > self.max_step:
                new = old + self.max_step
            elif delta < -self.max_step:
                new = old - self.max_step

            self.last[k] = new
            out[k] = clamp_0_100(new)

        return out


# ============================================================
# BrainCo adapter
# ============================================================

class BrainCoRevo1Adapter:
    """
    BrainCo Revo1 adapter.

    Important:
        All BrainCo async SDK calls must run on the same persistent event loop.
        Do NOT use asyncio.run() repeatedly, because it creates and closes a loop each time.
    """

    def __init__(self, dry_run=True, port="/dev/ttyUSB0", quick=True):
        self.dry_run = dry_run
        self.port = port
        self.quick = quick

        self.client = None
        self.slave_id = None
        self.loop = None
        self.closed = False

        if self.dry_run:
            print("[BrainCo] DRY-RUN mode. No command will be sent to the real hand.")
            return

        if open_modbus_revo1 is None:
            raise RuntimeError(
                "Cannot import revo1_utils. Please check SDK path:\n"
                f"  {DEFAULT_REVO1_DIR}"
            )

        print("[BrainCo] Connecting to Revo1 hand...")
        print(f"[BrainCo] Port: {self.port}")

        # Create one persistent event loop for all BrainCo SDK calls.
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.loop.run_until_complete(self._connect_async())

    async def _connect_async(self):
        self.client, self.slave_id = await open_modbus_revo1(
            port_name=self.port,
            quick=self.quick,
        )

        print(f"[BrainCo] Connected. slave_id={self.slave_id}")

        try:
            info = await self.client.get_device_info(self.slave_id)
            print("[BrainCo] Device info:", getattr(info, "description", info))
        except Exception as e:
            print("[BrainCo] Warning: failed to read device info:", e)

    async def _send_async(self, positions):
        await self.client.set_finger_positions(self.slave_id, positions)

    def send(self, cmd_1000):
        """
        cmd_1000:
            semantic command dict in official 0~1000 range.

        Official SDK:
            positions = [Thumb, ThumbAux, Index, Middle, Ring, Pinky]
            0=open, 1000=closed
        """
        positions = cmd_to_sdk_positions(cmd_1000)

        if self.dry_run:
            print(
                "[DRY-RUN SDK] "
                f"positions={positions} "
                f"(Thumb={positions[0]}, ThumbAux={positions[1]}, "
                f"Index={positions[2]}, Middle={positions[3]}, "
                f"Ring={positions[4]}, Pinky={positions[5]})"
            )
            return

        if self.closed:
            print("[BrainCo] Warning: send() called after adapter closed.")
            return

        self.loop.run_until_complete(self._send_async(positions))

    def safe_open(self):
        """
        Official SDK:
            0 = open
        """
        open_cmd_1000 = {
            "thumb_horizontal": 0,
            "thumb_vertical": 0,
            "index": 0,
            "middle": 0,
            "ring": 0,
            "little": 0,
        }
        self.send(open_cmd_1000)

    async def _close_async(self):
        try:
            print("[BrainCo] Sending open pose before closing...")
            positions = [0, 0, 0, 0, 0, 0]
            await self.client.set_finger_positions(self.slave_id, positions)
            await asyncio.sleep(0.5)
        except Exception as e:
            print("[BrainCo] Warning: failed to send open pose:", e)

        try:
            libstark.modbus_close(self.client)
            print("[BrainCo] Modbus client closed.")
        except Exception as e:
            print("[BrainCo] Warning: failed to close Modbus client:", e)

    def close(self):
        if self.dry_run or self.closed:
            return

        self.closed = True

        try:
            self.loop.run_until_complete(self._close_async())
        finally:
            try:
                self.loop.close()
            except Exception:
                pass


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bridge Quest HTS hand tracking to BrainCo Revo1 hand."
    )

    parser.add_argument(
        "--hand",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Quest hand to use. Default: left.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="HTS TCP server host. Default: 0.0.0.0.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTS TCP server port. Default: 8000.",
    )

    parser.add_argument(
        "--brainco-port",
        type=str,
        default="/dev/ttyUSB0",
        help="BrainCo serial port. Default: /dev/ttyUSB0.",
    )

    parser.add_argument(
        "--send-hz",
        type=float,
        default=5.0,
        help="Command sending frequency. Default: 5 Hz.",
    )

    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.30,
        help="EMA smoothing alpha in 0~100 space. Default: 0.30.",
    )

    parser.add_argument(
        "--max-step",
        type=float,
        default=3.0,
        help="Maximum command change per send in 0~100 space. Default: 3.",
    )

    parser.add_argument(
        "--enable-hand",
        action="store_true",
        help="Actually send commands to the real BrainCo hand. Default is dry-run.",
    )

    parser.add_argument(
        "--no-open-on-exit",
        action="store_true",
        help="Do not send open command when exiting.",
    )

    parser.add_argument(
        "--thumb-calib",
        type=str,
        default=None,
        help="Path to thumb horizontal calibration json.",
    )

    parser.add_argument(
        "--print-hz",
        type=float,
        default=5.0,
        help="Debug print frequency. Default: 5 Hz.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    dry_run = not args.enable_hand

    print("========================================")
    print("HTS → BrainCo Revo1 Bridge")
    print(f"HTS listening:   {args.host}:{args.port}")
    print(f"Quest hand:      {args.hand}")
    print(f"BrainCo port:    {args.brainco_port}")
    print(f"Send Hz:         {args.send_hz}")
    print(f"Smooth alpha:    {args.smooth_alpha}")
    print(f"Max step 0~100:  {args.max_step}")
    print(f"Max step 0~1000: {args.max_step * 10:.0f}")
    print(f"Dry run:         {dry_run}")
    print("")
    print("SDK convention:")
    print("  0    = open")
    print("  1000 = closed")
    print("")
    print("SDK order:")
    print("  [Thumb, ThumbAux, Index, Middle, Ring, Pinky]")
    print("========================================")
    print("")

    if not dry_run:
        print("WARNING: Real hand control is enabled.")
        print("Make sure the hand is free and emergency stop is available.")
        input("Press ENTER to continue, or Ctrl+C to abort...")

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host=args.host,
            port=args.port,
            timeout_s=1.0,
            output=StreamOutput.FRAMES,
        )
    )

    hand_adapter = BrainCoRevo1Adapter(
        dry_run=dry_run,
        port=args.brainco_port,
        quick=True,
    )
    thumb_calib = load_thumb_calibration(args.thumb_calib)

    smoother = CommandSmoother(alpha=args.smooth_alpha)
    limiter = StepLimiter100(max_step=args.max_step)

    send_interval = 1.0 / max(args.send_hz, 1e-6)
    last_send_time = 0.0
    last_print_time = 0.0
    print_interval = 1.0 / max(args.print_hz, 1e-6)

    try:
        print("Waiting for HTS hand frames...")
        print("Quest app should use:")
        print("  Protocol: TCP")
        print("  Host/IP:  127.0.0.1")
        print("  Port:     8000")
        print("")

        for event in client.iter_events():
            if not is_target_hand_frame(event, args.hand):
                continue

            now = time.time()
            if now - last_send_time < send_interval:
                continue
            last_send_time = now

            side_value = get_side_value(event)
            points = event.landmarks.points

            # 1. HTS landmarks -> normalized command, 0~100
            raw_cmd_100 = compute_brainco_0_100(points)

            if thumb_calib is not None:
                raw_cmd_100["thumb_horizontal"] = compute_thumb_horizontal_calibrated(
                    points,
                    thumb_calib,
                )

            # 2. Smooth in 0~100 space
            smooth_cmd_100 = smoother.update(raw_cmd_100)

            # 3. Step limit in 0~100 space
            safe_cmd_100 = limiter.update(smooth_cmd_100)

            # 4. Convert to official SDK 0~1000
            safe_cmd_1000 = cmd_100_to_cmd_1000(safe_cmd_100)

            positions = cmd_to_sdk_positions(safe_cmd_1000)
            do_print = (now - last_print_time) >= print_interval

            if do_print:
                last_print_time = now
                print(
                    f"{side_value.upper():5s} | "
                    f"0~100: "
                    f"thumb_h={safe_cmd_100['thumb_horizontal']:3d}, "
                    f"thumb_v={safe_cmd_100['thumb_vertical']:3d}, "
                    f"index={safe_cmd_100['index']:3d}, "
                    f"middle={safe_cmd_100['middle']:3d}, "
                    f"ring={safe_cmd_100['ring']:3d}, "
                    f"little={safe_cmd_100['little']:3d} "
                    f"|| SDK 0~1000 positions={positions}"
                )

            hand_adapter.send(safe_cmd_1000)

    except KeyboardInterrupt:
        print("")
        print("Ctrl+C received. Stopping bridge...")

        if not args.no_open_on_exit:
            try:
                hand_adapter.safe_open()
            except Exception as e:
                print("Failed to send safe open command:", e)

    finally:
        hand_adapter.close()
        print("Bridge stopped.")


if __name__ == "__main__":
    main()