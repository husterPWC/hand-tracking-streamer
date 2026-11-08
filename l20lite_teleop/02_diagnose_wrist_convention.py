#!/usr/bin/env python3
"""
第二阶段：WristPose quaternion ↔ landmark palm frame 的坐标 convention 诊断。

背景：
  PalmFrameTracker V2 假设「wrist quaternion 的相对旋转变化」等于「landmark
  palm frame 的相对旋转变化」。但 HTS wrist quaternion 与 landmark 的坐标
  convention 可能不一致（轴置换 / 轴反号 / 四元数方向约定相反）。
  当手腕翻转约 90° 时，err 会升到约 170°，说明存在一个固定的坐标变换没对齐。

方法：
  枚举 24 个 proper axis transform（octahedral 群，det=+1 的带符号置换矩阵） ×
  2 个四元数方向约定（正向 / 共轭反向），对每条轨迹统计 quat 搬运的 palm frame
  与 landmark palm frame 之间的 mean / median / max / p95 朝向误差，找到最优的
  固定坐标变换 S 与方向约定 h。

  对每个 (S, h)：
    ΔR_w^h(t) = R_w(t)·R_w(0)ᵀ          (h = +1, 正向)
              = R_w(t)ᵀ·R_w(0)          (h = -1, 反向/共轭)
    R_quat(t)  = S · ΔR_w^h(t) · Sᵀ · R_lm(0)
    err(t)     = angle(R_quat(t), R_lm(t))

  其中 R_lm(t) 由 landmark 构造（与 l20lite_hand_features._landmark_palm_frame 一致）。

用法：
  # 实时采集一段轨迹（务必包含手腕翻转 ~90° 的动作）后立即分析：
  python 02_diagnose_wrist_convention.py --hand right --seconds 20 --save wrist.npz

  # 离线分析已保存的轨迹（无需 Quest 连接）：
  python 02_diagnose_wrist_convention.py --load wrist.npz

  # 用合成数据自检枚举逻辑是否正确：
  python 02_diagnose_wrist_convention.py --selftest
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from l20lite_hand_features import (
    JOINT_INDEX,
    _as_points,
    _landmark_palm_frame,
)
from hand_tracking_sdk import (
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)
from hand_tracking_sdk.convert import convert_landmarks_unity_left_to_right


# ============================================================
# 24 个 proper axis transform（octahedral 群）
# ============================================================

def generate_proper_axis_transforms():
    """返回 24 个 det=+1 的 3×3 带符号置换矩阵（坐标轴 → 坐标轴的正交旋转变换）。"""
    mats = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3), dtype=np.int64)
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i, p] = s
            if np.linalg.det(M) > 0.0:
                mats.append(M)
    return mats


_AXIS_LABELS = ["x", "y", "z"]


def describe_axis_transform(S):
    """把变换 S 写成可读的轴映射：wrist 的 x/y/z 分别落到 landmark 的哪个轴。"""
    parts = []
    for j, name in enumerate(_AXIS_LABELS):
        v = S[:, j]  # wrist 轴 j 在 landmark 基下的像
        idx = int(np.argmax(np.abs(v)))
        sign = "" if v[idx] > 0 else "(flip)"
        parts.append(f"w_{name}→lm_{_AXIS_LABELS[idx]}{sign}")
    return ", ".join(parts)


# ============================================================
# 朝向误差
# ============================================================

def rotation_angle_deg(Ra, Rb):
    """两个旋转序列之间的角距离（度）。Ra/Rb: (..., 3, 3)。"""
    rel = np.einsum("...ij,...jk->...ik", np.swapaxes(Ra, -1, -2), Rb)
    trace = np.einsum("...ii->...", rel)
    cosang = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


# ============================================================
# 核心枚举
# ============================================================

def enumerate_conventions(R_lm_seq, R_w_seq, top=12):
    """
    枚举 48 个 (S, h) 组合，返回按 median 朝向误差升序排序的结果。

    R_lm_seq: (T, 3, 3) landmark palm frame 旋转矩阵序列
    R_w_seq : (T, 3, 3) wrist quaternion 旋转矩阵序列
    """
    R_lm_seq = np.asarray(R_lm_seq, dtype=np.float64)
    R_w_seq = np.asarray(R_w_seq, dtype=np.float64)
    T = R_lm_seq.shape[0]
    if T < 3:
        raise ValueError("轨迹帧数太少（< 3），无法诊断。")

    R_lm0 = R_lm_seq[0]
    R_w0 = R_w_seq[0]

    # 预计算两种方向约定的相对旋转（跳过 t=0）
    dR_plus = R_w_seq[1:] @ R_w0.T              # h=+1: R_w(t)·R_w(0)ᵀ
    dR_minus = np.swapaxes(R_w_seq[1:], -1, -2) @ R_w0  # h=-1: R_w(t)ᵀ·R_w(0)
    R_lm_tail = R_lm_seq[1:]

    results = []
    for S_int in generate_proper_axis_transforms():
        S = S_int.astype(np.float64)
        ST = S.T
        for h, dR in ((+1, dR_plus), (-1, dR_minus)):
            dR_lm = S @ dR @ ST                      # (T-1, 3, 3)
            R_quat = dR_lm @ R_lm0                   # (T-1, 3, 3)
            errs = rotation_angle_deg(R_quat, R_lm_tail)
            results.append({
                "S": S_int,
                "h": h,
                "mean": float(np.mean(errs)),
                "median": float(np.median(errs)),
                "max": float(np.max(errs)),
                "p95": float(np.percentile(errs, 95)),
                "errs": errs,
            })

    results.sort(key=lambda r: r["median"])
    return results


# ============================================================
# 数据采集
# ============================================================

def record_trajectory(client, hand, seconds):
    """
    从 HTS 采集 `seconds` 秒的 (landmarks_right_handed, wrist_quat_xyzw)。
    返回 (landmarks: (T,21,3), wrist_quat: (T,4), side)。
    """
    landmarks = []
    wrist_quats = []
    start = time.time()
    print(f"采集 {seconds}s（请做手腕翻转 ~90° 的动作）...", flush=True)

    for event in client.iter_events():
        side_obj = getattr(event, "side", None)
        side_key = getattr(side_obj, "value", str(side_obj))
        if hand != "both" and hand != side_key.lower():
            continue

        lm = convert_landmarks_unity_left_to_right(event.landmarks)
        pts = _as_points(lm.points)
        landmarks.append(pts)
        wrist_quats.append([
            event.wrist.qx, event.wrist.qy, event.wrist.qz, event.wrist.qw,
        ])

        if time.time() - start >= seconds:
            break

    landmarks = np.asarray(landmarks, dtype=np.float64)
    wrist_quats = np.asarray(wrist_quats, dtype=np.float64)
    print(f"采集完成：{landmarks.shape[0]} 帧。", flush=True)
    return landmarks, wrist_quats


# ============================================================
# 分析（landmarks → R_lm 序列，再枚举）
# ============================================================

def landmarks_to_Rlm(landmarks):
    """把 (T,21,3) 右手系 landmark 转成 (T,3,3) palm frame 旋转矩阵。"""
    T = landmarks.shape[0]
    R = np.empty((T, 3, 3), dtype=np.float64)
    for t in range(T):
        _, Rt = _landmark_palm_frame(landmarks[t])
        R[t] = Rt
    return R


def analyze(landmarks, wrist_quats, top=12):
    """主分析：把 wrist quaternion 转 R_w，构造 R_lm，跑枚举并打印。"""
    R_w = Rotation.from_quat(wrist_quats).as_matrix()   # (T,3,3)
    R_lm = landmarks_to_Rlm(landmarks)
    results = enumerate_conventions(R_lm, R_w, top=top)
    return results


# ============================================================
# 输出
# ============================================================

def print_report(results, top=12):
    print()
    print("=" * 96)
    print("WristPose quaternion ↔ landmark palm frame convention 诊断结果")
    print("按 median 朝向误差升序（单位：度）")
    print("=" * 96)
    header = (f"{'rank':>4}  {'h':>2}  {'mean':>7}  {'median':>7}  "
              f"{'max':>7}  {'p95':>7}  axis mapping")
    print(header)
    print("-" * 96)
    for i, r in enumerate(results[:top], start=1):
        print(f"{i:>4}  {r['h']:>+2}  {r['mean']:7.2f}  {r['median']:7.2f}  "
              f"{r['max']:7.2f}  {r['p95']:7.2f}  {describe_axis_transform(r['S'])}")
    print("-" * 96)

    best = results[0]
    print()
    print("推荐固定变换（最佳组合）：")
    print(f"  h = {best['h']:+d}  ({'正向' if best['h'] == 1 else '反向/共轭'})")
    print(f"  S =")
    for row in best["S"]:
        print(f"      {list(int(v) for v in row)}")
    print(f"  轴映射: {describe_axis_transform(best['S'])}")
    print(f"  误差: mean={best['mean']:.2f}°  median={best['median']:.2f}°  "
          f"max={best['max']:.2f}°  p95={best['p95']:.2f}°")
    print()

    # 对照：当前实现（S=I 的两种方向约定）
    eye = np.eye(3, dtype=np.int64)
    for r in results:
        if np.array_equal(r["S"], eye) and r["h"] == +1:
            cur_a = r
        if np.array_equal(r["S"], eye) and r["h"] == -1:
            cur_b = r
    print("对照（当前 PalmFrameTracker 的行为 = S=I）：")
    print(f"  Candidate A (h=+1): median={cur_a['median']:.2f}°  max={cur_a['max']:.2f}°")
    print(f"  Candidate B (h=-1): median={cur_b['median']:.2f}°  max={cur_b['max']:.2f}°")
    print()


# ============================================================
# 自检（合成数据，验证枚举逻辑能恢复已知变换）
# ============================================================

def self_test():
    rng = np.random.default_rng(0)
    transforms = generate_proper_axis_transforms()
    # 挑一个非平凡的真变换
    S_true = transforms[13].astype(np.float64)
    while np.array_equal(S_true, np.eye(3)):
        S_true = transforms[rng.integers(len(transforms))].astype(np.float64)
    h_true = -1
    T = 200

    qs = rng.normal(size=(T, 4))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    R_w = Rotation.from_quat(qs).as_matrix()

    R_lm0 = Rotation.random(random_state=1).as_matrix()
    R_lm = np.empty_like(R_w)
    R_lm[0] = R_lm0
    ST = S_true.T
    for t in range(T):
        if h_true == +1:
            d = R_w[t] @ R_w[0].T
        else:
            d = R_w[t].T @ R_w[0]
        R_lm[t] = S_true @ d @ ST @ R_lm0

    results = enumerate_conventions(R_lm, R_w, top=3)
    best = results[0]
    ok_S = np.array_equal(best["S"].astype(np.float64), S_true)
    ok_h = best["h"] == h_true
    ok_err = best["median"] < 1e-6

    print("自检：合成数据（已知 S_true / h_true =", describe_axis_transform(S_true),
          "/", h_true, "）")
    print("  枚举得到 top1：", describe_axis_transform(best["S"]), "/ h =", best["h"],
          "/ median =", best["median"], "°")
    if ok_S and ok_h and ok_err:
        print("  ✅ 自检通过：枚举正确恢复了已知变换。")
    else:
        print("  ❌ 自检失败：", dict(S=ok_S, h=ok_h, err=ok_err))
        sys.exit(1)


# ============================================================
# 保存 / 加载
# ============================================================

def save_npz(path, landmarks, wrist_quats, hand):
    np.savez_compressed(
        path,
        landmarks=landmarks.astype(np.float32),
        wrist_quat=wrist_quats.astype(np.float32),
        hand=np.asarray([hand]),
    )
    print(f"已保存轨迹到 {path}")


def load_npz(path):
    data = np.load(path, allow_pickle=False)
    return (
        data["landmarks"].astype(np.float64),
        data["wrist_quat"].astype(np.float64),
    )


# ============================================================
# 主入口
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="诊断 HTS wrist quaternion 坐标 convention。")
    p.add_argument("--hand", type=str, default="right", choices=["right", "left", "both"])
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--save", type=str, default=None, help="采集后另存 npz")
    p.add_argument("--load", type=str, default=None, help="离线分析 npz（不连 Quest）")
    p.add_argument("--selftest", action="store_true", help="用合成数据自检枚举逻辑")
    return p.parse_args()


def main():
    args = parse_args()

    if args.selftest:
        self_test()
        return

    if args.load:
        landmarks, wrist_quats = load_npz(args.load)
        print(f"加载轨迹：{landmarks.shape[0]} 帧")
    else:
        client = HTSClient(
            HTSClientConfig(
                transport_mode=TransportMode.TCP_SERVER,
                host=args.host,
                port=args.port,
                timeout_s=1.0,
                output=StreamOutput.FRAMES,
            )
        )
        try:
            landmarks, wrist_quats = record_trajectory(client, args.hand, args.seconds)
        except KeyboardInterrupt:
            print("\n采集被中断，用已采集帧分析。")
        if landmarks.shape[0] < 3:
            print("帧数不足，无法诊断。")
            return
        if args.save:
            save_npz(args.save, landmarks, wrist_quats, args.hand)

    results = analyze(landmarks, wrist_quats, top=args.top)
    print_report(results, top=args.top)


if __name__ == "__main__":
    main()
