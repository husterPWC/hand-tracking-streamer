#!/usr/bin/env python3
"""
第四阶段：个人标定。采集人手每个语义自由度的 open/closed 端点，生成 retargeter 用 JSON。

原理：
  retarget 里每个自由度都是「端点线性映射」：
    flex[f]    : [open, closed] → [100, 0]
    abd[f]     : [together, spread] → [0, 100]
    thumb_flex : chord_z（带符号）  [open, closed] → [100, 0]
    thumb_abd  : azim（方位角）      [open, closed] → [0, 100]
    thumb_yaw  : roll_deg           neutral（50 中立 + 增益）
  这些端点随人手型/朝向不同，需逐人采集。

流程：
  脚本引导你依次做 6 个姿势，每个姿势保持约 --seconds 秒，取该窗口内特征的中位数
  作为端点，最后写入 JSON。标定只需 HTS（不用连真机）。

运行前先连 Quest：
  cd ~/codePWC/hand-tracking-streamer && ./hts_vr_connect.sh

运行：
  python 04_calibrate.py --hand right --out calibration/right.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from hand_tracking_sdk import (
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)

from l20lite_hand_features import extract_features
from l20lite_retargeter import ABD_FINGERS, FLEX_FINGERS, thumb_proxies


# 每个姿势采集哪些量
POSES = [
    {
        "name": "OPEN",
        "desc": "四指伸直并拢，拇指自然放松（远离食指）",
        "seconds": 4.0,
        "keys": ["flex_open", "abd_together", "thumb_abd_open"],
    },
    {
        "name": "FIST",
        "desc": "四指握拳，尽量卷紧",
        "seconds": 4.0,
        "keys": ["flex_closed"],
    },
    {
        "name": "SPREAD",
        "desc": "四指伸直状态下张开到最大",
        "seconds": 4.0,
        "keys": ["abd_spread"],
    },
    {
        "name": "THUMB_UP",
        "desc": "竖大拇指 👍（拇指尽量远离掌心、向上伸）",
        "seconds": 4.0,
        "keys": ["thumb_flex_open"],
    },
    {
        "name": "THUMB_FLEX_IN",
        "desc": "拇指弯向掌心，指尖尽量贴近手掌",
        "seconds": 4.0,
        "keys": ["thumb_flex_closed"],
    },
    {
        "name": "THUMB_OPPOSE",
        "desc": "拇指对掌，指尖尽量碰到食指",
        "seconds": 4.0,
        "keys": ["thumb_abd_closed"],
    },
]


def _median_finite(values):
    """忽略 NaN 的中位数；全 NaN 时返回 0.0。"""
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return 0.0
    return float(np.median(vals))


def _collect(client, hand, seconds):
    """采集 `seconds` 秒的原始特征，返回各量中位数。"""
    flex = {f: [] for f in FLEX_FINGERS}
    abd_deg = {f: [] for f in ABD_FINGERS}
    chord_z = []
    azim = []
    roll_deg = []

    start = time.time()
    n = 0
    for event in client.iter_events():
        side_obj = getattr(event, "side", None)
        side_key = getattr(side_obj, "value", str(side_obj))
        if hand != "both" and hand != side_key.lower():
            continue

        feats = extract_features(event)
        for f in FLEX_FINGERS:
            flex[f].append(float(feats["flex_latents"][f]))
        for f in ABD_FINGERS:
            abd_deg[f].append(float(np.degrees(feats["abd_angles"][f])))
        cz, az, rd = thumb_proxies(feats)
        chord_z.append(cz)
        azim.append(az)
        roll_deg.append(rd)
        n += 1

        if time.time() - start >= seconds:
            break

    if n == 0:
        raise RuntimeError("未采集到任何帧，请确认 HTS 连接与 hand 参数。")

    return {
        "flex": {f: float(np.median(flex[f])) for f in FLEX_FINGERS},
        "abd_deg": {f: float(np.median(abd_deg[f])) for f in ABD_FINGERS},
        "chord_z": float(np.median(chord_z)),
        "azim": float(np.median(azim)),
        "roll_deg": _median_finite(roll_deg),
        "n": n,
    }


def parse_args():
    p = argparse.ArgumentParser(description="L20Lite 个人标定采集。")
    p.add_argument("--hand", type=str, default="right", choices=["right", "left", "both"])
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seconds", type=float, default=4.0, help="每个姿势采集时长")
    p.add_argument("--out", type=str, default=None, help="输出 JSON 路径")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or f"calibration/{args.hand}.json"

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host=args.host,
            port=args.port,
            timeout_s=1.0,
            output=StreamOutput.FRAMES,
        )
    )

    print("=" * 72)
    print("L20Lite 个人标定采集")
    print(f"hand: {args.hand}   每个姿势 {args.seconds}s")
    print("=" * 72)
    print()

    results = {}
    try:
        for i, pose in enumerate(POSES, start=1):
            print(f"[{i}/{len(POSES)}] {pose['name']} — {pose['desc']}")
            print(f"        准备好后按 Enter 开始采集 {args.seconds}s ...")
            input("        > ")
            data = _collect(client, args.hand, args.seconds)
            results[pose["name"]] = data
            print(f"        ✓ 采集 {data['n']} 帧")
            print(f"          flex={ {f: round(data['flex'][f],3) for f in FLEX_FINGERS} }")
            print(f"          abd_deg={ {f: round(data['abd_deg'][f],1) for f in ABD_FINGERS} }")
            print(f"          chord_z={data['chord_z']:.3f}  azim={data['azim']:.1f}°  roll_deg={data['roll_deg']:.1f}°")
            print()
    except KeyboardInterrupt:
        print("\n标定中断。")
        return

    # 组装 calib JSON
    calib = {
        "flex": {
            f: {
                "open": results["OPEN"]["flex"][f],
                "closed": results["FIST"]["flex"][f],
            }
            for f in FLEX_FINGERS
        },
        "abd": {
            f: {
                "together": results["OPEN"]["abd_deg"][f],
                "spread": results["SPREAD"]["abd_deg"][f],
            }
            for f in ABD_FINGERS
        },
        "thumb_flex": {
            "open": results["THUMB_UP"]["chord_z"],
            "closed": results["THUMB_FLEX_IN"]["chord_z"],
        },
        "thumb_abd": {
            "open": results["OPEN"]["azim"],
            "closed": results["THUMB_FLEX_IN"]["azim"],
        },
        "thumb_yaw": {
            "neutral": results["OPEN"]["roll_deg"],
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    print("=" * 72)
    print(f"标定已保存到 {out_path}")
    print("=" * 72)
    print(json.dumps(calib, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
