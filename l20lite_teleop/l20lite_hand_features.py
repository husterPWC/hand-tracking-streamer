"""
L20Lite 人手特征提取模块（纯 HTS，不依赖 mujoco/linkerbot）。

从 HandFrame 的 21×3 landmarks 提取人手语义特征：

  1. Palm Local Frame（直接由 landmark 构造，天然手腕锚定，无需 wrist quaternion）
  2. 四指 flex latent（MCP/PIP/DIP 三个弯曲角 → L20Lite 机械 synergy 投影）
  3. 三指 ABD signed angle（palm plane 里的侧摆角）
  4. thumb 几何观测（TIP 位置 / MCP→TIP 方向 / thumb-index 距离）

关键结论（实测验证）：
  HTS 的 landmarks 本身就在「手腕局部坐标系」里——wrist landmark 恒为原点，
  四个 MCP 基本不随手腕旋转而移动，只有指尖随手指弯曲而动。因此直接用
  landmark 构造的 palm frame 在手腕旋转时天然稳定，不需要、也不该用 wrist
  quaternion 的相对旋转去搬运它（那样反而会把 palm frame 转飞，产生 90°/170°
  级别的假 err）。wrist quaternion 是手在世界系的朝向，与 10-D 手指语义
  （flex/abd/thumb）无关，故本模块不使用它。

HTS landmark 索引（21 个）：
  0  wrist
  1  thumb_CMC   2  thumb_MCP   3  thumb_IP   4  thumb_TIP
  5  index_MCP   6  index_PIP   7  index_DIP   8  index_TIP
  9  middle_MCP 10  middle_PIP 11  middle_DIP 12  middle_TIP
 13  ring_MCP   14  ring_PIP   15  ring_DIP   16  ring_TIP
 17  little_MCP 18  little_PIP 19  little_DIP 20  little_TIP
"""

from __future__ import annotations

import numpy as np
from hand_tracking_sdk.convert import convert_landmarks_unity_left_to_right


# ============================================================
# Landmark 索引
# ============================================================

JOINT_INDEX = {
    "wrist": 0,
    "thumb_cmc": 1,
    "thumb_mcp": 2,
    "thumb_ip": 3,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_pip": 6,
    "index_dip": 7,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_dip": 11,
    "middle_tip": 12,
    "ring_mcp": 13,
    "ring_pip": 14,
    "ring_dip": 15,
    "ring_tip": 16,
    "little_mcp": 17,
    "little_pip": 18,
    "little_dip": 19,
    "little_tip": 20,
}


# ============================================================
# 手指链（MCP / PIP / DIP / TIP 四个点）
# ============================================================

FINGER_CHAINS = {
    "index": ["index_mcp", "index_pip", "index_dip", "index_tip"],
    "middle": ["middle_mcp", "middle_pip", "middle_dip", "middle_tip"],
    "ring": ["ring_mcp", "ring_pip", "ring_dip", "ring_tip"],
    # L20Lite SDK 语义叫 pinky，HTS landmark 叫 little
    "pinky": ["little_mcp", "little_pip", "little_dip", "little_tip"],
}


# ============================================================
# L20Lite 四指 flex 机械 synergy（mechanical coupling）
# ============================================================

FLEX_SYNERGY = {
    "index": np.array([1.0, 1.30, 0.46], dtype=np.float64),
    "middle": np.array([1.0, 1.30, 0.46], dtype=np.float64),
    "ring": np.array([1.0, 1.34, 0.46], dtype=np.float64),
    "pinky": np.array([1.0, 1.34, 0.46], dtype=np.float64),
}


# ============================================================
# 基础几何工具
# ============================================================

def _as_points(points):
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if arr.shape[0] < 21:
        raise ValueError(f"Expected at least 21 landmarks, got {arr.shape[0]}")
    return arr


def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _angle_between(v1, v2):
    v1 = _norm(np.asarray(v1, dtype=np.float64))
    v2 = _norm(np.asarray(v2, dtype=np.float64))
    cosang = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.arccos(cosang))


def _bend_angle(points, i_a, i_b, i_c):
    """三点 i_a - i_b - i_c 在 i_b 处的弯曲角（伸直=0，弯曲=大，弧度）。"""
    v1 = points[i_a] - points[i_b]
    v2 = points[i_c] - points[i_b]
    return float(np.pi - _angle_between(v1, v2))


# ============================================================
# Palm Local Frame
# ============================================================

def _landmark_palm_frame(points):
    """
    用 landmark 直接构造 palm frame（唯一来源，不再有 quaternion 搬运）。

    origin = 四个 MCP 的几何中心
    forward = wrist → middle MCP
    lateral = index MCP → pinky MCP（Gram-Schmidt 正交化）
    normal  = forward × lateral
    """
    wrist = points[JOINT_INDEX["wrist"]]
    index_mcp = points[JOINT_INDEX["index_mcp"]]
    middle_mcp = points[JOINT_INDEX["middle_mcp"]]
    ring_mcp = points[JOINT_INDEX["ring_mcp"]]
    little_mcp = points[JOINT_INDEX["little_mcp"]]

    origin = (index_mcp + middle_mcp + ring_mcp + little_mcp) / 4.0

    forward = _norm(middle_mcp - wrist)
    lateral_raw = little_mcp - index_mcp
    lateral = lateral_raw - np.dot(lateral_raw, forward) * forward
    lateral = _norm(lateral)
    normal = _norm(np.cross(forward, lateral))

    R = np.column_stack([forward, lateral, normal])
    return origin, R


def to_palm_frame(points, origin, R):
    """把所有 landmark 转换到 palm local frame（相对 origin，R^T 旋转）。"""
    return (points - origin) @ R


# ============================================================
# 四指关节弯曲角
# ============================================================

def compute_finger_bend_angles(points_palm, finger):
    """
    计算某根手指的 MCP/PIP/DIP 三个弯曲角（伸直=0，弯曲=大，弧度）。

    返回 [θ_MCP, θ_PIP, θ_DIP]。
    """
    mcp, pip, dip, tip = [JOINT_INDEX[n] for n in FINGER_CHAINS[finger]]

    theta_pip = _bend_angle(points_palm, mcp, pip, dip)
    theta_dip = _bend_angle(points_palm, pip, dip, tip)

    wrist_idx = JOINT_INDEX["wrist"]
    proximal_dir = _norm(points_palm[pip] - points_palm[mcp])
    metacarpal_dir = _norm(points_palm[mcp] - points_palm[wrist_idx])
    theta_mcp = _angle_between(proximal_dir, metacarpal_dir)

    return np.array([theta_mcp, theta_pip, theta_dip], dtype=np.float64)


# ============================================================
# 四指 flex latent（synergy 投影）
# ============================================================

def compute_flex_latent(points_palm, finger, weight=None):
    """
    q* = (bᵀ W θ) / (bᵀ W b)，把 [θ_MCP,θ_PIP,θ_DIP] 投影到 synergy b。
    """
    theta = compute_finger_bend_angles(points_palm, finger)
    b = FLEX_SYNERGY[finger]

    if weight is None:
        w = np.ones(3, dtype=np.float64)
    else:
        w = np.asarray(weight, dtype=np.float64).reshape(-1)
        if w.shape != (3,):
            raise ValueError("weight must have length 3")

    num = float(np.dot(b * w, theta))
    den = float(np.dot(b * w, b))
    if abs(den) < 1e-9:
        return 0.0
    return num / den


# ============================================================
# 三指 ABD signed angle
# ============================================================

def compute_abd_angle(points_palm, finger, ref_finger):
    """
    某根手指相对参考手指的侧摆 signed angle（palm plane 里，绕 normal 轴）。
    """
    mcp = JOINT_INDEX[FINGER_CHAINS[finger][0]]
    pip = JOINT_INDEX[FINGER_CHAINS[finger][1]]
    ref_mcp = JOINT_INDEX[FINGER_CHAINS[ref_finger][0]]
    ref_pip = JOINT_INDEX[FINGER_CHAINS[ref_finger][1]]

    d = _norm(points_palm[pip] - points_palm[mcp])
    d_ref = _norm(points_palm[ref_pip] - points_palm[ref_mcp])

    n = np.array([0.0, 0.0, 1.0])  # palm frame z 轴

    d_plane = _norm(d - np.dot(d, n) * n)
    d_ref_plane = _norm(d_ref - np.dot(d_ref, n) * n)

    cross = np.cross(d_ref_plane, d_plane)
    dot = float(np.clip(np.dot(d_ref_plane, d_plane), -1.0, 1.0))
    angle = float(np.arctan2(float(np.dot(cross, n)), dot))
    return angle


# ============================================================
# thumb 几何观测
# ============================================================

def compute_thumb_observations(points_palm):
    thumb_tip = points_palm[JOINT_INDEX["thumb_tip"]]
    thumb_mcp = points_palm[JOINT_INDEX["thumb_mcp"]]
    index_tip = points_palm[JOINT_INDEX["index_tip"]]
    index_mcp = points_palm[JOINT_INDEX["index_mcp"]]

    axis = _norm(thumb_tip - thumb_mcp)

    return {
        "tip_position": thumb_tip.copy(),
        "axis_direction": axis.copy(),
        "tip_index_distance": float(np.linalg.norm(thumb_tip - index_tip)),
        "thumb_index_dist": float(np.linalg.norm(thumb_tip - index_mcp)),
    }


# ============================================================
# 总入口
# ============================================================

def extract_features(frame):
    """
    从 HandFrame 提取所有 human features。

    frame 需要：
      frame.landmarks.points -> 21×3 list

    返回 dict：
      palm_frame_origin / palm_frame_R
      finger_bend_angles / flex_latents / abd_angles / thumb / palm_width
    """
    # landmark 转右手系（HTS landmark 是 Unity 左手系，cross 需右手系）。
    # landmarks 本身在手腕局部系（wrist 恒为原点），直接构造 palm frame
    # 即可，无需 wrist quaternion。
    landmarks = convert_landmarks_unity_left_to_right(frame.landmarks)

    points = _as_points(landmarks.points)

    origin, R = _landmark_palm_frame(points)
    points_palm = to_palm_frame(points, origin, R)

    index_mcp = points_palm[JOINT_INDEX["index_mcp"]]
    little_mcp = points_palm[JOINT_INDEX["little_mcp"]]
    palm_width = float(np.linalg.norm(index_mcp - little_mcp))

    finger_bend = {}
    flex_latents = {}
    for finger in FINGER_CHAINS:
        finger_bend[finger] = compute_finger_bend_angles(points_palm, finger)
        flex_latents[finger] = compute_flex_latent(points_palm, finger)

    abd_angles = {
        "index": compute_abd_angle(points_palm, "index", "middle"),
        "ring": compute_abd_angle(points_palm, "ring", "middle"),
        "pinky": compute_abd_angle(points_palm, "pinky", "ring"),
    }

    thumb = compute_thumb_observations(points_palm)

    return {
        "palm_frame_origin": origin,
        "palm_frame_R": R,
        "points_palm": points_palm,
        "palm_width": palm_width,
        "finger_bend_angles": finger_bend,
        "flex_latents": flex_latents,
        "abd_angles": abd_angles,
        "thumb": thumb,
    }
