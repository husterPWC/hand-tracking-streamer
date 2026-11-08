#!/usr/bin/env python3
"""
拇指特征调试：打印 dist_palm / chord_z(带符号) / dist_index / roll_deg。

用于确定 thumb_flex 到底该用「距离 dist_palm」还是「方向 chord_z」。
关键验证：拇指「垂直掌心」和「平放贴食指」两个姿势，看 chord_z 能否分开。

请做三个姿势，观察 chord_z 这列：
  1) 垂直掌心（👍，拇指竖起来）→ chord_z 应该明显大（接近 ±1）
  2) 平放贴食指（5 指同一平面）→ chord_z 应该接近 0
  3) 弯进掌心 → chord_z 应该小/负

运行前先连 Quest：./hts_vr_connect.sh
运行：python 05_thumb_debug.py --hand right --hz 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from hand_tracking_sdk import (
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)

from l20lite_hand_features import JOINT_INDEX, extract_features


def thumb_proxies_debug(features):
    pts = features["points_palm"]
    w = features["palm_width"]
    if w < 1e-6:
        w = 1.0
    mcp = pts[JOINT_INDEX["thumb_mcp"]]
    ip = pts[JOINT_INDEX["thumb_ip"]]
    tip = pts[JOINT_INDEX["thumb_tip"]]
    index_mcp = pts[JOINT_INDEX["index_mcp"]]

    dist_palm = float(np.linalg.norm(tip)) / w
    dist_index = float(np.linalg.norm(tip - index_mcp)) / w

    chord = tip - mcp
    cn = float(np.linalg.norm(chord))
    if cn < 1e-9:
        cn = 1.0
    chord_z = float(chord[2]) / cn
    azim = float(np.degrees(np.arctan2(chord[1], chord[0])))

    cmc = pts[JOINT_INDEX["thumb_cmc"]]
    v_meta = mcp - cmc
    v_prox = ip - mcp
    mn = float(np.linalg.norm(v_meta))
    pn = float(np.linalg.norm(v_prox))
    roll = float("nan")
    if mn > 1e-9 and pn > 1e-9:
        v_meta = v_meta / mn
        v_prox = v_prox / pn
        z = np.array([0.0, 0.0, 1.0])
        ref = z - np.dot(z, v_meta) * v_meta
        rn = float(np.linalg.norm(ref))
        prox_perp = v_prox - np.dot(v_prox, v_meta) * v_meta
        pn2 = float(np.linalg.norm(prox_perp))
        if rn > 1e-9 and pn2 > 1e-9:
            ref = ref / rn
            prox_perp = prox_perp / pn2
            cross = np.cross(ref, prox_perp)
            roll = float(np.degrees(np.arctan2(float(np.dot(cross, v_meta)),
                                               float(np.dot(ref, prox_perp)))))
    return dist_palm, chord_z, azim, dist_index, roll


def parse_args():
    p = argparse.ArgumentParser(description="拇指特征调试。")
    p.add_argument("--hand", type=str, default="right", choices=["right", "left", "both"])
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--hz", type=float, default=5.0)
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
    last_print = 0.0

    print("=" * 90)
    print("拇指特征调试。做「远离食指 / 靠近食指」两个姿势，看 azim 能不能分开。")
    print("=" * 90)
    print(f"{'dist_palm':>10s}  {'chord_z':>8s}  {'azim':>8s}  {'dist_index':>11s}  {'roll_deg':>10s}")
    print("-" * 90)

    try:
        for event in client.iter_events():
            side_obj = getattr(event, "side", None)
            side_key = getattr(side_obj, "value", str(side_obj))
            if args.hand != "both" and args.hand != side_key.lower():
                continue

            now = time.time()
            if now - last_print < print_interval:
                continue
            last_print = now

            feats = extract_features(event)
            dp, cz, az, di, rd = thumb_proxies_debug(feats)
            rd_s = "  nan " if np.isnan(rd) else f"{rd:+8.1f}°"

            print(f"{dp:10.3f}  {cz:+8.3f}  {az:+8.1f}  {di:11.3f}  {rd_s:>10s}")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
