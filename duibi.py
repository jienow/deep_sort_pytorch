import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

###############################################
# 1. 读取 MOT 格式 result.txt
###############################################

def load_tracks(path):
    """
    读取你的 result.txt，返回 dict:
    track_id → list of (frame_id, bbox)
    """
    tracks = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            frame = int(items[0])
            tid   = int(items[1])
            x, y, w, h = map(float, items[3:7])
            tracks[tid].append((frame, (x, y, w, h)))
    return tracks


###############################################
# 2. FRAG 计算（轨迹断裂次数）
###############################################

def compute_frag(track_dict):
    frag = 0
    for tid, entries in track_dict.items():
        frames = sorted([f for f, _ in entries])
        for i in range(1, len(frames)):
            if frames[i] != frames[i - 1] + 1:
                frag += 1
    return frag


###############################################
# 3. 计算 IOU（供 IDSW 使用）
###############################################

def bbox_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


###############################################
# 4. IDSW 计算（基于邻帧 IOU 判断 ID 是否切换）
###############################################

def compute_idsw(track_dict):
    # 按帧组织
    frame_dict = defaultdict(list)
    for tid, entries in track_dict.items():
        for frame, bbox in entries:
            frame_dict[frame].append((tid, bbox))

    frames = sorted(frame_dict.keys())
    idsw = 0

    for i in range(1, len(frames)):
        prev = frame_dict[frames[i - 1]]
        curr = frame_dict[frames[i]]

        for tid1, bbox1 in prev:
            best_iou, best_tid = 0, None
            for tid2, bbox2 in curr:
                iou = bbox_iou(bbox1, bbox2)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid2

            # IOU > 0.3 认为是同一个目标，但 ID 换了
            if best_iou > 0.3 and best_tid != tid1:
                idsw += 1

    return idsw


###############################################
# 5. 画图函数（柱状图）
###############################################

def plot_metric(title, baseline_value, improved_value, ylabel):
    plt.figure(figsize=(5, 4))
    methods = ['Baseline', 'Improved']
    values = [baseline_value, improved_value]
    colors = ['#d9534f', '#5cb85c']  # 红=差, 绿=好

    plt.bar(methods, values, color=colors)
    plt.ylabel(ylabel)
    plt.title(title)

    # 在柱子上标数字
    for i, v in enumerate(values):
        plt.text(i, v + 0.05 * max(values), str(v), ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()


###############################################
# 6. 主程序：载入、计算、绘图
###############################################

baseline_path = "result_baseline.txt"
improved_path = "result_improved.txt"

baseline = load_tracks(baseline_path)
improved = load_tracks(improved_path)

# FRAG
frag_base = compute_frag(baseline)
frag_impr = compute_frag(improved)

# IDSW
idsw_base = compute_idsw(baseline)
idsw_impr = compute_idsw(improved)

print("======== 结果输出 ========")
print("Baseline FRAG:", frag_base)
print("Improved FRAG:", frag_impr)
print("-------------------------")
print("Baseline IDSW:", idsw_base)
print("Improved IDSW:", idsw_impr)

# 绘图
plot_metric("轨迹连续性（FRAG）对比", frag_base, frag_impr, "FRAG（越低越好）")
plot_metric("ID 切换次数（IDSW）对比", idsw_base, idsw_impr, "IDSW（越低越好）")
