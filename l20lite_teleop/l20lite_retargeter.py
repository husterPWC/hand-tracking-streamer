"""
L20Lite Semantic Retargeter。

把人手 10 个语义特征（来自 l20lite_hand_features.extract_features）映射到
L20Lite 10-D SDK 命令（0~100，Real SDK semantic 顺序）。

输出顺序（与 linkerbot / l20lite_joint_mapping.py 一致）：
  [0] thumb_flex
  [1] thumb_abd
  [2] index_flex
  [3] middle_flex
  [4] ring_flex
  [5] pinky_flex
  [6] index_abd
  [7] ring_abd
  [8] pinky_abd
  [9] thumb_yaw

方向约定（真机实测，见 project notes §16）：
  flex       : 100 ≈ 伸直，数值↓ = 更弯
  abd        : 数值↑ = 远离参考指（张开）
  thumb_flex : 数值↓ = 拇指向掌心弯曲（100 = 远离掌心/竖起）
  thumb_abd  : 数值↑ = 拇指靠近食指（对掌）
  thumb_yaw  : 数值↑ = 顺时针旋转

拇指说明：
  thumb_flex 用「方向量」chord_z（能区分垂直/平放/弯掌心）：
    thumb_flex : chord(MCP→TIP) 的带符号 z 分量 —— 垂直≈+0.7，平放≈0，弯掌心≈-0.1
  thumb_abd 用「方向量」azim（thumb chord 在掌心平面的方位角，敏感于左右摆）：
    thumb_abd  : atan2(chord_y, chord_x)   —— 远离食指≈负，靠近食指≈正
  thumb_yaw 用「旋转量」（近节指骨 MCP→IP 绕掌骨 CMC→MCP 的方位角），与弯曲正交：
    roll = 近节指骨方向在「垂直于掌骨」平面内的方位角（相对掌心法向）。

映射（端点线性 + 夹取，端点由 calib 提供）：
  flex[f]    : flex_latent(rad)  [open, closed] → [100, 0]
  abd[f]     : abd_angle(deg)    [together, spread] → [0, 100]
  thumb_flex : chord_z（带符号）  [open, closed] → [100, 0]
  thumb_abd  : azim（方位角）     [open, closed] → [0, 100]
  thumb_yaw  : roll_deg          50 + SIGN·GAIN·(roll_deg − neutral)，夹取到 [0,100]

calib 不传时用 DEFAULT_CALIB；个人标定后由 04_calibrate.py 生成 JSON，用
load_calib 载入并传给 retarget(features, calib)。
"""

from __future__ import annotations

import json

import numpy as np

from l20lite_hand_features import JOINT_INDEX


# ============================================================
# 输出顺序
# ============================================================

L20_ORDER = [
    "thumb_flex",
    "thumb_abd",
    "index_flex",
    "middle_flex",
    "ring_flex",
    "pinky_flex",
    "index_abd",
    "ring_abd",
    "pinky_abd",
    "thumb_yaw",
]

FLEX_FINGERS = ["index", "middle", "ring", "pinky"]
ABD_FINGERS = ["index", "ring", "pinky"]

# thumb_yaw：旋转方位角变化 → 命令（50 为中立）。方向反了翻 SIGN，幅度不对调 GAIN。
THUMB_YAW_SIGN = +1.0
THUMB_YAW_GAIN = 2.5

# thumb_flex 死区：chord_z 低于该值都算「平放/弯掌心」（thumb_flex=0），
# 只有明显竖起来（chord_z 高于该值）才抬升。解决「平放贴食指仍翘起」的问题。
FLAT_CHORD_Z = 0.40

# thumb_yaw 固定偏移：真机拇指整体偏小指一侧（旋转方向偏了）。
# 加偏移往食指方向拧。符号/大小看实测调（正值或负值都试）。
THUMB_YAW_OFFSET = 30.0


# ============================================================
# 默认标定（无个人标定时的占位端点）
# ============================================================

def _default_calib():
    return {
        "flex": {f: {"open": 0.15, "closed": 1.20} for f in FLEX_FINGERS},
        "abd": {f: {"together": -15.0, "spread": 15.0} for f in ABD_FINGERS},
        "thumb_flex": {"open": 0.70, "closed": -0.10},  # chord_z（带符号）：垂直≈+0.7，弯掌心≈-0.1
        "thumb_abd": {"open": -40.0, "closed": +40.0},  # azim（度）：远离≈负，靠近≈正
        "thumb_yaw": {"neutral": 0.0},                   # roll（度）
    }


def load_calib(path):
    """从 JSON 载入个人标定（04_calibrate.py 生成）。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 工具
# ============================================================

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _clamp100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def _linear(x, x_open, x_closed, out_open, out_closed):
    """x_open→out_open, x_closed→out_closed，线性夹取到 [out_open, out_closed]。"""
    denom = x_closed - x_open
    if abs(denom) < 1e-9:
        return float(out_open)
    t = _clamp01((x - x_open) / denom)
    return out_open + t * (out_closed - out_open)


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


# ============================================================
# thumb 几何代理
# ============================================================

def thumb_roll_deg(features):
    """拇指旋转角（度）：近节指骨 MCP→IP 绕掌骨 CMC→MCP 的方位角。

    与弯曲正交（弯曲改极角、旋转改方位角）。拇指伸直（近节指骨∥掌骨）时退化，
    返回 NaN。
    """
    pts = features["points_palm"]
    cmc = pts[JOINT_INDEX["thumb_cmc"]]
    mcp = pts[JOINT_INDEX["thumb_mcp"]]
    ip = pts[JOINT_INDEX["thumb_ip"]]

    v_meta = mcp - cmc
    v_prox = ip - mcp
    v_meta = _norm(v_meta)
    v_prox = _norm(v_prox)
    if v_meta is None or v_prox is None:
        return float("nan")

    # 参考方向：掌心法向（z）投影到垂直于掌骨的平面
    z = np.array([0.0, 0.0, 1.0])
    ref = z - np.dot(z, v_meta) * v_meta
    ref = _norm(ref)
    if ref is None:
        return float("nan")

    # 近节指骨投影到垂直于掌骨的平面
    prox_perp = v_prox - np.dot(v_prox, v_meta) * v_meta
    prox_perp = _norm(prox_perp)
    if prox_perp is None:
        return float("nan")  # 拇指完全伸直，旋转无定义

    cross = np.cross(ref, prox_perp)
    dot = float(np.clip(np.dot(ref, prox_perp), -1.0, 1.0))
    roll = float(np.degrees(np.arctan2(float(np.dot(cross, v_meta)), dot)))
    return roll


def thumb_proxies(features):
    """返回 (chord_z, azim, roll_deg)。

    chord_z    : thumb chord(MCP→TIP) 在掌心法向的「带符号」z 分量 ∈ [-1,1]。
                 垂直掌心≈+0.7，平放贴食指≈0，弯进掌心≈-0.1（带符号，不能取 abs）。
    azim       : thumb chord 在掌心平面的方位角（度）。远离食指≈负，靠近食指≈正。
                 用于 thumb_abd（左右摆），比距离量敏感得多。
    roll_deg   : 近节指骨绕掌骨轴的方位角（旋转，与弯曲正交）
    """
    pts = features["points_palm"]
    mcp = pts[JOINT_INDEX["thumb_mcp"]]
    tip = pts[JOINT_INDEX["thumb_tip"]]

    chord = tip - mcp
    chord_n = float(np.linalg.norm(chord))
    if chord_n < 1e-9:
        chord_n = 1.0
    chord_z = float(chord[2]) / chord_n
    azim = float(np.degrees(np.arctan2(chord[1], chord[0])))

    roll_deg = thumb_roll_deg(features)
    return chord_z, azim, roll_deg


# ============================================================
# 总入口
# ============================================================

def retarget(features, calib=None) -> np.ndarray:
    """人手语义特征 → L20Lite 10-D SDK 命令（0~100），返回 np.ndarray(10)。"""
    if calib is None:
        calib = _default_calib()

    flex = features["flex_latents"]
    abd = features["abd_angles"]
    chord_z, azim, roll_deg = thumb_proxies(features)

    out = {}

    # thumb
    out["thumb_flex"] = _linear(
        chord_z, calib["thumb_flex"]["open"], calib["thumb_flex"]["closed"], 100.0, 0.0,
    )
    out["thumb_abd"] = _linear(
        azim, calib["thumb_abd"]["open"], calib["thumb_abd"]["closed"], 100.0, 0.0,
    )
    # thumb_yaw：旋转方位角（拇指伸直退化时保持中立）
    if np.isnan(roll_deg):
        out["thumb_yaw"] = 50.0
    else:
        out["thumb_yaw"] = _clamp100(
            50.0 + THUMB_YAW_SIGN * THUMB_YAW_GAIN * (roll_deg - calib["thumb_yaw"]["neutral"])
            + THUMB_YAW_OFFSET
        )

    # 四指 flex
    for f in FLEX_FINGERS:
        c = calib["flex"][f]
        out[f"{f}_flex"] = _linear(float(flex[f]), c["open"], c["closed"], 100.0, 0.0)

    # 三指 abd
    for f in ABD_FINGERS:
        c = calib["abd"][f]
        deg = float(np.degrees(abd[f]))
        out[f"{f}_abd"] = _linear(deg, c["together"], c["spread"], 0.0, 100.0)

    return np.array([out[k] for k in L20_ORDER], dtype=np.float64)


def retarget_dict(features, calib=None) -> dict:
    """同 retarget，但返回按名字索引的 dict（调试/打印用）。"""
    vec = retarget(features, calib=calib)
    return {name: float(v) for name, v in zip(L20_ORDER, vec)}
