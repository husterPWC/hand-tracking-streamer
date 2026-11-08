#!/usr/bin/env python3
"""
真机大拇指三自由度单独控制（绕过 HTS/retargeter，直接发命令）。

用于逐个 debug 每个自由度到底让拇指怎么动：
  [0] thumb_flex（flex）：弯曲伸直（卷 ↔ 翘）
  [1] thumb_abd  (abd) ：左右摆（远离 ↔ 靠近食指）
  [9] thumb_yaw  (yaw) ：旋转（顺 ↔ 逆时针）

用法（先配好 can0，无需连 Quest）：
  # 交互式
  python 06_thumb_manual.py --hand right

  # 自动扫一遍某个自由度（0→100→0，每个值停约 1 秒）
  python 06_thumb_manual.py --hand right --sweep flex
  python 06_thumb_manual.py --hand right --sweep all

交互命令：
  flex 50 / abd 80 / yaw 20   设置某个自由度（0~100，其它保持中立）
  sweep flex / sweep all       扫一遍看动作
  read                        读当前 10 个角度
  open                        回中立姿态
  quit                        退出（退出前自动回中立）
"""

from __future__ import annotations

import argparse
import time

from linkerbot import L20lite
from linkerbot.exceptions import TimeoutError


ORDER = [
    "thumb_flex", "thumb_abd", "index_flex", "middle_flex", "ring_flex",
    "pinky_flex", "index_abd", "ring_abd", "pinky_abd", "thumb_yaw",
]

# 基础姿态：拇指三自由度中立(50)、四指伸直(100)、三指 abd 中立(50)
BASE = [50.0, 50.0, 100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 50.0]

# 通道名 → 索引（含短别名）
CHANNEL = {
    "flex": 0, "f": 0,
    "abd": 1, "a": 1,
    "yaw": 9, "y": 9,
}


def read_angles(hand):
    try:
        data = hand.angle.get_blocking(timeout_ms=1000)
        return data.angles.to_list()
    except TimeoutError:
        return None


def print_angles(q):
    print("  " + " ".join(f"{n}={v:5.1f}" for n, v in zip(ORDER, q)))


def sweep(hand, name):
    jid = CHANNEL[name]
    for v in (0, 25, 50, 75, 100, 75, 50, 25, 0):
        q = list(BASE)
        q[jid] = float(v)
        hand.angle.set_angles(q)
        time.sleep(1.2)
        cur = read_angles(hand)
        cur_v = cur[jid] if cur else float("nan")
        print(f"  [{name}] target={v:3d}  readback={cur_v:5.1f}")


def parse_args():
    p = argparse.ArgumentParser(description="真机拇指三自由度单独控制。")
    p.add_argument("--hand", type=str, default="right", choices=["right", "left"])
    p.add_argument("--can-interface", type=str, default="can0")
    p.add_argument("--sweep", type=str, default=None,
                   choices=["flex", "abd", "yaw", "all"], help="自动扫一遍")
    return p.parse_args()


def main():
    args = parse_args()

    with L20lite(side=args.hand, interface_name=args.can_interface) as hand:
        print("已连接真机。发送中立姿态。")
        hand.angle.set_angles(BASE)
        time.sleep(0.5)

        if args.sweep:
            names = ["flex", "abd", "yaw"] if args.sweep == "all" else [args.sweep]
            for n in names:
                print(f"\n=== sweep {n} ===")
                sweep(hand, n)
            print("\n回到中立姿态。")
            hand.angle.set_angles(BASE)
            return

        print("\n命令：flex/abd/yaw <0~100> | sweep <flex|abd|yaw|all> | read | open | quit\n")

        q = list(BASE)
        while True:
            try:
                cmd = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd:
                continue
            parts = cmd.split()
            c = parts[0].lower()

            if c in ("quit", "q", "exit"):
                break
            elif c in ("open", "o", "reset"):
                q = list(BASE)
                hand.angle.set_angles(q)
                print("  已回中立姿态")
            elif c in ("read", "r"):
                cur = read_angles(hand)
                print_angles(cur) if cur else print("  读取超时")
            elif c in ("sweep", "s"):
                name = parts[1].lower() if len(parts) > 1 else "all"
                if name == "all":
                    for n in ("flex", "abd", "yaw"):
                        sweep(hand, n)
                elif name in CHANNEL:
                    sweep(hand, name)
                else:
                    print("  未知自由度:", name)
            elif c in CHANNEL:
                try:
                    v = float(parts[1])
                except (IndexError, ValueError):
                    print("  用法：flex 50")
                    continue
                v = max(0.0, min(100.0, v))
                q[CHANNEL[c]] = v
                hand.angle.set_angles(q)
                cur = read_angles(hand)
                cur_v = cur[CHANNEL[c]] if cur else float("nan")
                print(f"  已设置 {c}={v:.0f}  (readback={cur_v:.1f})")
            else:
                print("  未知命令:", c)

        hand.angle.set_angles(BASE)
        print("已回中立姿态，退出。")


if __name__ == "__main__":
    main()
