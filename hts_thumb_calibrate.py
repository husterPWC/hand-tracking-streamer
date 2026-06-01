import argparse
import json
import time
import numpy as np

from hand_tracking_sdk import HTSClient, HTSClientConfig, StreamOutput, TransportMode


def np_point(p):
    return np.array(p, dtype=np.float32)


def norm(v):
    return float(np.linalg.norm(v))


def get_side_value(event):
    side = getattr(event, "side", None)
    return getattr(side, "value", str(side))


def is_target_hand_frame(event, target_hand):
    cls_name = type(event).__name__
    side_value = get_side_value(event)

    if cls_name != "HandFrame":
        return False

    if target_hand == "left":
        return side_value == "Left"

    if target_hand == "right":
        return side_value == "Right"

    return False


def palm_features(points):
    """
    Return candidate raw features for thumb-horizontal calibration.

    The calibration script will automatically choose the feature that changes most
    between your open and inward-thumb poses.
    """
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

    features = {
        # thumb tip closer to palm center
        "dist_thumb_palm": norm(thumb_tip - palm_center) / palm_width,

        # thumb tip closer to index side
        "dist_thumb_index": norm(thumb_tip - index_base) / palm_width,

        # thumb tip closer to middle base / palm centerline
        "dist_thumb_middle": norm(thumb_tip - middle_base) / palm_width,

        # projection along palm left-right axis
        "proj_across": float(np.dot(rel, across_axis)) / palm_width,

        # projection along palm normal
        "proj_normal": float(np.dot(rel, palm_normal)) / palm_width,
    }

    return features


def capture_pose(client, target_hand, seconds, label):
    print("")
    print("========================================")
    print(f"Prepare pose: {label}")
    print("Hold your hand steady in front of Quest.")
    input("Press ENTER to start capture...")
    print(f"Capturing for {seconds:.1f} seconds...")

    values = []
    start = time.time()

    for event in client.iter_events():
        if not is_target_hand_frame(event, target_hand):
            continue

        points = event.landmarks.points
        values.append(palm_features(points))

        if time.time() - start >= seconds:
            break

    if not values:
        raise RuntimeError("No hand frames captured. Check Quest streaming and hand side.")

    keys = values[0].keys()
    mean_features = {}

    for k in keys:
        arr = np.array([v[k] for v in values], dtype=np.float32)
        mean_features[k] = float(np.mean(arr))

    print(f"Captured {len(values)} frames.")
    print("Mean raw features:")
    for k, v in mean_features.items():
        print(f"  {k}: {v:.4f}")

    return mean_features


def choose_best_feature(open_features, closed_features):
    best_key = None
    best_delta = -1.0

    for k in open_features.keys():
        delta = abs(closed_features[k] - open_features[k])
        if delta > best_delta:
            best_delta = delta
            best_key = k

    return best_key, best_delta


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hand",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Which hand to calibrate. Default: left.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=2.0,
        help="Capture seconds for each pose. Default: 2.0.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output calibration json path.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output = args.output
    if output is None:
        output = f"thumb_horizontal_calib_{args.hand}.json"

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host=args.host,
            port=args.port,
            timeout_s=1.0,
            output=StreamOutput.FRAMES,
        )
    )

    print("========================================")
    print("Thumb Horizontal Calibration")
    print(f"Target hand: {args.hand}")
    print(f"Listening: {args.host}:{args.port}")
    print("")
    print("Quest app:")
    print("  Protocol: TCP")
    print("  Host/IP:  127.0.0.1")
    print(f"  Port:     {args.port}")
    print("========================================")

    open_features = capture_pose(
        client,
        args.hand,
        args.seconds,
        "thumb horizontal OPEN / thumb away from palm",
    )

    closed_features = capture_pose(
        client,
        args.hand,
        args.seconds,
        "thumb horizontal CLOSED / thumb moves toward palm",
    )

    best_key, best_delta = choose_best_feature(open_features, closed_features)

    calib = {
        "hand": args.hand,
        "channel": "thumb_horizontal",
        "feature": best_key,
        "open": open_features[best_key],
        "closed": closed_features[best_key],
        "delta": best_delta,
        "open_features": open_features,
        "closed_features": closed_features,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

    print("")
    print("========================================")
    print("Calibration saved.")
    print(f"Output: {output}")
    print(f"Best feature: {best_key}")
    print(f"Open raw:   {calib['open']:.4f}")
    print(f"Closed raw: {calib['closed']:.4f}")
    print(f"Delta:      {best_delta:.4f}")
    print("========================================")


if __name__ == "__main__":
    main()

'''
它会让你做两个动作：
1. 大拇指横向张开 / 远离手心
2. 大拇指横向往手心收
'''
