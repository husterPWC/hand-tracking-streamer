当前目标：
HTS Quest3 → L20Lite 10-D 遥操作

当前目录：
/home/robot/codePWC/hand-tracking-streamer

核心目录：
l20lite/
l20lite_teleop/

运行环境（重要）：
- HTS 特征提取 / 诊断脚本（l20lite_teleop/*）：用 conda env `hts`
  （有 hand_tracking_sdk + numpy/scipy，无 mujoco）。
- MuJoCo 仿真（l20lite/*）：用 conda env `openarmx_hts_mj`
  （唯一装了 mujoco 3.6.0 的环境）。
- 两个环境都保留，勿删。

已经完成：
1. L20Lite Sim-Real 10-D 验证
2. HTS 21×3 landmarks 读取
3. 四指 flex synergy feature
4. 三指 ABD feature
5. thumb geometry feature
6. ✅ Palm Frame / WristPose 问题已解决（见下）

已解决的关键结论（wrist quaternion 问题）：
- HTS 的 landmarks 本身就在「手腕局部坐标系」里：wrist landmark 恒为原点，
  四个 MCP 不随手腕旋转移动，只有指尖随手指弯曲而动。
- 因此 landmark 直接构造的 palm frame 在手腕旋转时天然稳定，根本不需要
  wrist quaternion 的相对旋转去搬运（搬运反而把 palm frame 转飞，产生
  90°/170° 假 err）。wrist quaternion 是手在世界系的朝向，与 10-D 手指
  语义无关，故弃用。
- 已改 l20lite_hand_features.py：删除 PalmFrameTracker / _slerp /
  _rot_angle_between / palm_err_deg，extract_features 直接用
  _landmark_palm_frame(points)。01_inspect 同步去掉 err 显示。
- 诊断工具 02_diagnose_wrist_convention.py 保留（--selftest ✅），
  用于复核/复现该结论（48 个 (S,h) 全部打平 = landmark 恒定、quaternion 独转）。

下一步（特征提取 + Retargeter 已就绪，继续闭环）：
1. ✅ l20lite_retargeter.py 已写好：extract_features → 10-D SDK 命令（0~100）。
   - 4 指 flex：flex_latent(rad) → 100(伸)~0(卷)，FLEX_OPEN/CLOSED 默认 0.15/1.20。
   - 3 指 abd：signed angle → 50(中立)+2.0*deg，ABD_SIGN 可翻方向。
   - thumb_flex：tip→palm center 距离 → 100(远)~0(近掌心)。
   - thumb_abd：tip→index_mcp 距离 → 0(远)~100(对掌)。
   - thumb_yaw：占位 50（thumb rotation 特征未实现，见 notes §37.3）。
2. ✅ 03_realtime_teleop.py 已写好（HTS → extract_features → retarget → 真机）。
   - 默认 dry-run（只打印），--enable-hand 才发真机；含 EMA 平滑 + step limit + 退出张开。
   - 真机 SDK 已装：linkerbot 0.5.4 装进 `hts` env（+pydantic/python-can）。
   - PCAN-USB 已识别（0c72:000c），can0 存在但当前 DOWN。
3. 下一步（用户手动）：
   a. sudo 配置 can0（bitrate 1 Mbps），见 03 文件头或 notes。
   b. cd l20lite_teleop && hts 环境跑 03_realtime_teleop.py --hand right（dry-run）
      确认 HTS→retarget 命令合理；再 --enable-hand 上真机。
4. 个人标定：真机 teleop 跑通后再做 open/closed 端点标定。
