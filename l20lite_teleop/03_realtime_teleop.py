#!/usr/bin/env python3
"""
第三阶段：HTS → 人手语义特征 → Semantic Retargeter → 真机 L20Lite 遥操作。

链路：
  Quest 3 → HTS (TCP 127.0.0.1:8000) → hand_tracking_sdk
    → extract_features() → 10 个人手语义特征
    → retarget() → 10-D SDK 命令 (0~100)
    → linkerbot.L20lite.angle.set_angles() → CAN → 真机

默认 dry-run（只打印命令，不发真机）。加 --enable-hand 才会真正控制手。

运行前：
  1. 建立 Quest 连接：cd ~/codePWC/hand-tracking-streamer && ./hts_vr_connect.sh
  2. 配置 CAN（需 sudo，见下方说明或项目 notes）：
       sudo ip link set can0 down 2>/dev/null || true
       sudo ip link set can0 type can bitrate 1000000 restart-ms 100
       sudo ip link set can0 txqueuelen 1024
       sudo ip link set can0 up

用法：
  # 先 dry-run 看命令（不发真机）：
  python 03_realtime_teleop.py --hand right

  # 真正控制真机：
  python 03_realtime_teleop.py --hand right --enable-hand
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

from l20lite_hand_features import extract_features
from l20lite_retargeter import L20_ORDER, load_calib, retarget

try:
    from linkerbot import L20lite
except Exception:  # linkerbot 未安装 / can 后端缺失
    L20lite = None


# 安全张开姿态（0~100）：thumb_flex=100(远离掌心)、thumb_abd=100(靠近食指)、四指伸直、三指 abd 中立、thumb_yaw=100
OPEN_POSE = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 100.0]


class Smoother:
    """0~100 空间的 EMA 平滑。"""

    def __init__(self, alpha=0.30):
        self.alpha = alpha
        self.state = None

    def update(self, v):
        v = np.asarray(v, dtype=np.float64)
        if self.state is None:
            self.state = v.copy()
        else:
            self.state = self.alpha * v + (1.0 - self.alpha) * self.state
        return self.state.copy()


class StepLimiter:
    """每次发送最多变化 max_step（0~100 空间），防突变。"""

    def __init__(self, max_step=3.0):
        self.max_step = max_step
        self.last = None

    def update(self, v):
        v = np.asarray(v, dtype=np.float64)
        if self.last is None:
            self.last = v.copy()
            return v.copy()
        d = np.clip(v - self.last, -self.max_step, self.max_step)
        self.last = self.last + d
        return self.last.copy()


def _fmt_vec(vec):
    return " ".join(f"{n}={v:5.1f}" for n, v in zip(L20_ORDER, vec))


def parse_args():
    p = argparse.ArgumentParser(description="HTS → L20Lite 真机遥操作。")
    p.add_argument("--hand", type=str, default="right", choices=["right", "left", "both"])
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--can-interface", type=str, default="can0")
    p.add_argument("--send-hz", type=float, default=20.0)
    p.add_argument("--print-hz", type=float, default=5.0)
    p.add_argument("--smooth-alpha", type=float, default=0.30)
    p.add_argument("--max-step", type=float, default=3.0)
    p.add_argument("--enable-hand", action="store_true", help="真正发送到真机（默认 dry-run）")
    p.add_argument("--readback", action="store_true", help="周期读取真机实际角度打印")
    p.add_argument("--no-open-on-exit", action="store_true", help="退出时不发张开姿态")
    p.add_argument("--calib", type=str, default=None, help="个人标定 JSON 路径（04_calibrate.py 生成）")
    return p.parse_args()


def main():
    args = parse_args()
    dry_run = not args.enable_hand

    if not dry_run and L20lite is None:
        print("错误：linkerbot 未安装，无法控制真机。请先 `pip install linkerbot`。")
        return

    calib = None
    if args.calib:
        calib = load_calib(args.calib)
        print(f"已加载个人标定：{args.calib}")

    print("=" * 88)
    print("HTS → L20Lite 实时遥操作 (第三阶段)")
    print(f"HTS: {args.host}:{args.port}  hand: {args.hand}")
    print(f"CAN: {args.can_interface}   send: {args.send_hz} Hz")
    print(f"dry-run: {dry_run}   smooth alpha: {args.smooth_alpha}   max step: {args.max_step}")
    print("输出顺序:", " ".join(L20_ORDER))
    print("=" * 88)
    print()

    if not dry_run:
        print("警告：真实手控制已启用，确保手处于自由状态且有急停手段。")
        print("退出时自动发送张开姿态。")

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host=args.host,
            port=args.port,
            timeout_s=1.0,
            output=StreamOutput.FRAMES,
        )
    )

    hand = None
    if not dry_run:
        hand = L20lite(side=args.hand, interface_name=args.can_interface)
        # 先发一个安全张开姿态
        hand.angle.set_angles(OPEN_POSE)
        print("真机已连接，已发送初始张开姿态。")

    smoother = Smoother(alpha=args.smooth_alpha)
    limiter = StepLimiter(max_step=args.max_step)

    send_interval = 1.0 / max(args.send_hz, 1e-6)
    print_interval = 1.0 / max(args.print_hz, 1e-6)
    last_send = 0.0
    last_print = 0.0

    def send(vec):
        if hand is not None:
            hand.angle.set_angles(vec.tolist())

    try:
        for event in client.iter_events():
            side_obj = getattr(event, "side", None)
            side_key = getattr(side_obj, "value", str(side_obj))
            if args.hand != "both" and args.hand != side_key.lower():
                continue

            now = time.time()
            if now - last_send < send_interval:
                continue
            last_send = now

            feats = extract_features(event)
            raw = retarget(feats, calib=calib)
            smooth = smoother.update(raw)
            limited = limiter.update(smooth)
            cmd = np.clip(limited, 0.0, 100.0)

            send(cmd)

            if now - last_print >= print_interval:
                last_print = now
                line = f"[{side_key.upper():5s}] cmd | {_fmt_vec(cmd)}"
                if args.readback and hand is not None:
                    try:
                        data = hand.angle.get_blocking(timeout_ms=200)
                        actual = data.angles.to_list()
                        line += "\n" + " " * 13 + "act | " + _fmt_vec(actual)
                    except Exception as e:
                        line += f"\n" + " " * 13 + f"(readback 失败: {e})"
                print(line)

    except KeyboardInterrupt:
        print("\nCtrl+C，停止。")

        if not dry_run and not args.no_open_on_exit and hand is not None:
            try:
                hand.angle.set_angles(OPEN_POSE)
                print("已发送张开姿态。")
            except Exception as e:
                print("发送张开姿态失败:", e)

    finally:
        if hand is not None:
            try:
                hand.close()
            except Exception:
                pass
        print("遥操作结束。")


if __name__ == "__main__":
    main()
