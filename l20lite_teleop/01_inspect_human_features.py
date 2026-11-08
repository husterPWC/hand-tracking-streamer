#!/usr/bin/env python3
"""
第一阶段：检查人手特征提取（不控制机器人）。

从 HTS 拿 HandFrame，提取人手语义特征并打印，验证：
  1. Palm Local Frame 是否稳定
  2. 四指 flex latent 是否随弯曲连续变化
  3. 三指 ABD signed angle 是否随张开连续变化
  4. thumb 几何观测是否合理

运行前先建立 Quest 连接：
  cd ~/codePWC/hand-tracking-streamer
  ./hts_vr_connect.sh

然后：
  python 01_inspect_human_features.py --hand right --hz 10
"""

import argparse
import sys
import time

from hand_tracking_sdk import (
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)

from l20lite_hand_features import extract_features


def parse_args():
    p = argparse.ArgumentParser(description="Inspect human hand features for L20Lite.")
    p.add_argument("--hand", type=str, default="right", choices=["right", "left", "both"])
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--hz", type=float, default=10.0)
    return p.parse_args()


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

    print_interval = 1.0 / max(args.hz, 1e-6)
    last_print = {"Right": 0.0, "Left": 0.0}

    print("=" * 72)
    print("L20Lite Human Feature Inspector (第一阶段)")
    print(f"Listening TCP {args.host}:{args.port}, target hand: {args.hand}")
    print()
    print("特征说明:")
    print("  flex latent : 四指弯曲程度（synergy 投影，伸直≈0，弯曲≈大，弧度）")
    print("  abd angle   : 三指侧摆 signed angle（弧度，正负表方向）")
    print("  thumb tip   : palm frame 里归一化坐标（除以 palm_width）")
    print("  thumb axis  : thumb MCP→TIP 单位方向")
    print("  dist        : thumb-index 距离（归一化）")
    print("=" * 72)
    print()

    try:
        for event in client.iter_events():
            # HandSide 是 StrEnum，要用 .value 拿 "Right"/"Left"，
            # str() 返回的是 "HandSide.RIGHT"
            side_obj = getattr(event, "side", None)
            side_key = getattr(side_obj, "value", str(side_obj))

            if args.hand != "both" and args.hand != side_key.lower():
                continue

            now = time.time()
            if now - last_print.get(side_key, 0.0) < print_interval:
                continue
            last_print[side_key] = now

            feats = extract_features(event)

            # 四指 flex latent（弧度）
            flex = feats["flex_latents"]
            # 三指 ABD（弧度）
            abd = feats["abd_angles"]
            # thumb 观测（归一化）
            w = feats["palm_width"]
            if w < 1e-6:
                w = 1.0
            tip = feats["thumb"]["tip_position"] / w
            axis = feats["thumb"]["axis_direction"]
            dist = feats["thumb"]["tip_index_distance"] / w

            def _deg(r):
                return round(float(r) * 180.0 / 3.14159, 1)

            print(
                f"[{side_key.upper():5s}] "
                f"flex[i={_deg(flex['index']):6.1f}° "
                f"m={_deg(flex['middle']):6.1f}° "
                f"r={_deg(flex['ring']):6.1f}° "
                f"p={_deg(flex['pinky']):6.1f}°] "
                f"abd[i={_deg(abd['index']):+6.1f}° "
                f"r={_deg(abd['ring']):+6.1f}° "
                f"p={_deg(abd['pinky']):+6.1f}°] "
                f"tip=({tip[0]:+.2f},{tip[1]:+.2f},{tip[2]:+.2f}) "
                f"dist={dist:.2f}"
            )

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
