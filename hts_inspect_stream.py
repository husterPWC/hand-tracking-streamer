from hand_tracking_sdk import HTSClient, HTSClientConfig, StreamOutput, TransportMode
import numpy as np
import time
import argparse


# =========================
# HTS landmark index
# =========================
#
# 0  Wrist
# 1  ThumbMetacarpal
# 2  ThumbProximal
# 3  ThumbDistal
# 4  ThumbTip
# 5  IndexProximal
# 6  IndexIntermediate
# 7  IndexDistal
# 8  IndexTip
# 9  MiddleProximal
# 10 MiddleIntermediate
# 11 MiddleDistal
# 12 MiddleTip
# 13 RingProximal
# 14 RingIntermediate
# 15 RingDistal
# 16 RingTip
# 17 LittleProximal
# 18 LittleIntermediate
# 19 LittleDistal
# 20 LittleTip

FINGER_CHAINS = {
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "little": [17, 18, 19, 20],
}


# =========================
# Math helpers
# =========================

def np_point(p):
    return np.array(p, dtype=np.float32)


def dist(a, b):
    a = np_point(a)
    b = np_point(b)
    return float(np.linalg.norm(a - b))


def angle_deg(a, b, c):
    """
    Compute angle a-b-c at point b.

    Straight finger joint:
        angle close to 180 deg.

    Bent finger joint:
        angle becomes smaller.
    """
    a = np_point(a)
    b = np_point(b)
    c = np_point(c)

    v1 = a - b
    v2 = c - b

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0

    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = np.clip(cosang, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosang)))


def clamp_0_100(x):
    return int(round(float(np.clip(x, 0.0, 100.0))))


def map_range_to_100(x, x_open, x_closed):
    """
    Map x to 0~100.

    x_open   -> 0
    x_closed -> 100

    Supports both increasing and decreasing ranges.
    """
    denom = x_closed - x_open
    if abs(denom) < 1e-6:
        return 0

    value = (x - x_open) / denom * 100.0
    return clamp_0_100(value)


# =========================
# Finger mapping
# =========================

def compute_normal_finger_curl(points, ids):
    """
    For index/middle/ring/little.

    Use two inner joint angles:
        p0-p1-p2
        p1-p2-p3

    Output:
        0   = open
        100 = closed
    """
    p0, p1, p2, p3 = [points[i] for i in ids]

    a1 = angle_deg(p0, p1, p2)
    a2 = angle_deg(p1, p2, p3)
    angle_sum = a1 + a2

    # Rough default values. Later we can replace them with calibration.
    open_angle_sum = 340.0
    closed_angle_sum = 190.0

    return map_range_to_100(
        x=angle_sum,
        x_open=open_angle_sum,
        x_closed=closed_angle_sum,
    )


def compute_thumb_vertical(points):
    """
    BrainCo thumb vertical channel.

    Strong BrainCo definition from user:
        thumb_vertical = thumb moves toward index finger.

    Human-hand approximation:
        Use the distance between thumb tip and index base.

    When thumb is away from index:
        distance is larger -> smaller control value.

    When thumb moves toward index:
        distance is smaller -> larger control value.

    Output:
        0   = thumb away from index
        100 = thumb toward index
    """
    thumb_tip = points[4]
    index_base = points[5]
    little_base = points[17]

    # Scale normalization by palm width.
    palm_width = dist(index_base, little_base)
    if palm_width < 1e-6:
        palm_width = 0.08

    d_thumb_index = dist(thumb_tip, index_base)
    d_norm = d_thumb_index / palm_width

    # Rough defaults:
    #   away from index: d_norm larger
    #   toward index:    d_norm smaller
    open_d = 1.45
    closed_d = 0.65

    return map_range_to_100(
        x=d_norm,
        x_open=open_d,
        x_closed=closed_d,
    )

def compute_thumb_horizontal(points):
    """
    BrainCo thumb horizontal channel.

    Strong BrainCo definition from user:
        thumb_horizontal = thumb moves toward palm center.

    Human-hand approximation:
        Use the distance between thumb tip and palm center.

    When thumb is open/outside:
        thumb tip is farther from palm center -> smaller control value.

    When thumb moves into palm:
        thumb tip is closer to palm center -> larger control value.

    Output:
        0   = thumb away from palm / open
        100 = thumb toward palm / inward
    """
    thumb_tip = points[4]

    wrist = points[0]
    index_base = points[5]
    middle_base = points[9]
    ring_base = points[13]
    little_base = points[17]

    # Palm center approximation.
    palm_center = (
        np_point(wrist)
        + np_point(index_base)
        + np_point(middle_base)
        + np_point(ring_base)
        + np_point(little_base)
    ) / 5.0

    # Scale normalization by palm width.
    palm_width = dist(index_base, little_base)
    if palm_width < 1e-6:
        palm_width = 0.08

    d_thumb_palm = dist(thumb_tip, palm_center)
    d_norm = d_thumb_palm / palm_width

    # Rough defaults:
    #   thumb open/outside: d_norm larger
    #   thumb inward to palm: d_norm smaller
    #
    # These values may need calibration after watching real output.
    open_d = 1.35
    closed_d = 0.55

    return map_range_to_100(
        x=d_norm,
        x_open=open_d,
        x_closed=closed_d,
    )


def compute_brainco_0_100(points):
    """
    Convert HTS 21 landmarks to BrainCo-style 0~100 command values.

    Output:
        thumb_horizontal: 0~100
        thumb_vertical:   0~100
        index:            0~100
        middle:           0~100
        ring:             0~100
        little:           0~100
    """
    cmd = {}

    cmd["thumb_horizontal"] = compute_thumb_horizontal(points)
    cmd["thumb_vertical"] = compute_thumb_vertical(points)

    for name, ids in FINGER_CHAINS.items():
        cmd[name] = compute_normal_finger_curl(points, ids)

    return cmd


# =========================
# Smoothing
# =========================

class CommandSmoother:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.state = None

    def update(self, cmd):
        if self.state is None:
            self.state = {k: float(v) for k, v in cmd.items()}
            return {k: clamp_0_100(v) for k, v in self.state.items()}

        for k, v in cmd.items():
            old = self.state.get(k, float(v))
            new = self.alpha * float(v) + (1.0 - self.alpha) * old
            self.state[k] = new

        return {k: clamp_0_100(v) for k, v in self.state.items()}


# =========================
# Event helpers
# =========================

def get_side_value(event):
    side = getattr(event, "side", None)
    return getattr(side, "value", str(side))


def is_target_hand_frame(event, target_hand):
    cls_name = type(event).__name__
    side_value = get_side_value(event)

    if cls_name != "HandFrame":
        return False

    if target_hand == "both":
        return side_value in ["Right", "Left"]

    if target_hand == "right":
        return side_value == "Right"

    if target_hand == "left":
        return side_value == "Left"

    return False


def print_raw_summary(event):
    side_value = get_side_value(event)
    points = event.landmarks.points

    print("")
    print(f"[RAW SUMMARY] {side_value}")
    print(f"  wrist pose:       {event.wrist}")
    print(f"  wrist landmark[0]: {points[0]}")
    print(f"  thumb tip[4]:      {points[4]}")
    print(f"  index tip[8]:      {points[8]}")
    print(f"  middle tip[12]:    {points[12]}")
    print(f"  ring tip[16]:      {points[16]}")
    print(f"  little tip[20]:    {points[20]}")
    print("")


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect HTS hand stream and convert landmarks to BrainCo-style 0~100 finger commands."
    )

    parser.add_argument(
        "--hand",
        type=str,
        default="right",
        choices=["right", "left", "both"],
        help="Which hand to print: right, left, or both. Default: right.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="TCP server host. Default: 0.0.0.0.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP server port. Default: 8000.",
    )

    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Print frequency per hand. Default: 10 Hz.",
    )

    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.35,
        help="EMA smoothing alpha. Larger is faster, smaller is smoother. Default: 0.35.",
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw landmark summary occasionally.",
    )

    return parser.parse_args()


# =========================
# Main
# =========================

def main():
    args = parse_args()

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host=args.host,
            port=args.port,
            timeout_s=1.0,
            output=StreamOutput.FRAMES,
        )
    )

    smoother_by_side = {
        "Right": CommandSmoother(alpha=args.smooth_alpha),
        "Left": CommandSmoother(alpha=args.smooth_alpha),
    }

    last_print_time_by_side = {
        "Right": 0.0,
        "Left": 0.0,
    }

    print_interval = 1.0 / max(args.hz, 1e-6)
    frame_count_by_side = {
        "Right": 0,
        "Left": 0,
    }

    print("========================================")
    print("HTS Inspect Stream")
    print(f"Listening on TCP {args.host}:{args.port} ...")
    print(f"Target hand: {args.hand}")
    print("")
    print("Quest app setting:")
    print("  Protocol: TCP")
    print("  Host/IP:  127.0.0.1")
    print(f"  Port:     {args.port}")
    print("")
    print("Output range:")
    print("  0   = open")
    print("  100 = closed / flexed")
    print("========================================")
    print("")

    for event in client.iter_events():
        if not is_target_hand_frame(event, args.hand):
            continue

        side_value = get_side_value(event)
        now = time.time()

        if now - last_print_time_by_side.get(side_value, 0.0) < print_interval:
            continue

        last_print_time_by_side[side_value] = now

        points = event.landmarks.points
        raw_cmd = compute_brainco_0_100(points)
        smooth_cmd = smoother_by_side[side_value].update(raw_cmd)

        frame_count_by_side[side_value] += 1

        print(
            f"{side_value.upper():5s} | "
            f"thumb_horizontal={smooth_cmd['thumb_horizontal']:3d} | "
            f"thumb_vertical={smooth_cmd['thumb_vertical']:3d} | "
            f"index={smooth_cmd['index']:3d} | "
            f"middle={smooth_cmd['middle']:3d} | "
            f"ring={smooth_cmd['ring']:3d} | "
            f"little={smooth_cmd['little']:3d}"
        )

        if args.raw and frame_count_by_side[side_value] % 30 == 0:
            print_raw_summary(event)


if __name__ == "__main__":
    main()


"""
运行方式:
只看右手
python hts_inspect_stream.py --hand right
只看左手
python hts_inspect_stream.py --hand left
左右手都看
python hts_inspect_stream.py --hand both
改打印频率，例如 20 Hz
python hts_inspect_stream.py --hand right --hz 20
打印原始 landmark 摘要
python hts_inspect_stream.py --hand right --raw
"""