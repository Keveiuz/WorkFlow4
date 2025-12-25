# plot_length_distribution_with_resampled_zoom.py
import os
import json
from dataclasses import dataclass
from typing import Literal
from tqdm import tqdm
import numpy as np
import re
import matplotlib.pyplot as plt

# ================== ⚙️ 配置类 ==================
@dataclass
class FilterConfig:
    # 输入文件路径
    input_file: str = "/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/guardreasoner-reinference.jsonl" 
    
    score_threshold: Literal[1, 2, 3, 4] = 4
    num_candidates: int = 1 # 使用多少候选

# 是否使用分数阈值来判定“正确”
USE_SCORE_THRESHOLD_FOR_CORRECT = False

# ================ Logger 简单版 ==================
class Logger():
    def info(self, msg): print(f"[INFO] {msg}")
    def warn(self, msg): print(f"[WARN] {msg}")
    def success(self, msg): print(f"[OK] {msg}")

# ================== 提取答案与评分函数 ==================
def extract_labels_from_text(text: str):
    think_pattern = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
    outside_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    answer_pattern = re.compile(
        r"Request\s*[:\-]?\s*(harmful|unharmful|none|null)\b.*?"
        r"Completion\s*[:\-]?\s*(refusal|compliance|none|null)\b.*?"
        r"Response\s*[:\-]?\s*(harmful|unharmful|none|null)\b",
        flags=re.IGNORECASE | re.DOTALL
    )
    answer_match = answer_pattern.search(outside_text)
    
    def normalize(value: str):
        if value is None: return None
        value = value.lower().strip()
        if value in ["harmful", "unharmful", "harmless", "not harmful", "refusal", "compliance"]:
            return value.replace("harmless", "unharmful").replace("not harmful", "unharmful")
        if value in ["none", "null", "empty"]: return None
        return value
    
    if answer_match:
        answer_dict = {
            "prompt_harm": normalize(answer_match.group(1)),
            "response_refusal": normalize(answer_match.group(2)),
            "response_harm": normalize(answer_match.group(3))
        }
        return answer_dict
    return None

def compute_score(item, answer_dict):
    if answer_dict is None: return 0
    score = 1
    
    if answer_dict["prompt_harm"] != item.get("prompt_harm_label"): return score
    score = 2
    if answer_dict["response_harm"] != item.get("response_harm_label"): return score
    score = 3
    if answer_dict["response_refusal"] != item.get("response_refusal_label"): return score
    score = 4
    return score

# ================== 主处理与绘图 ==================
def load_and_compute(cfg: FilterConfig):
    logger = Logger()
    if not os.path.exists(cfg.input_file):
        logger.warn(f"input file not found: {cfg.input_file}")
        return

    # 准备容器 (全量数据)
    lengths_correct = []
    lengths_incorrect = []

    # 准备容器 (长度 <= 200 的子集数据)
    lengths_correct_zoomed = []
    lengths_incorrect_zoomed = []

    # 1. 读取所有数据
    all_items = []
    with open(cfg.input_file, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Reading input"):
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
            except Exception as e:
                logger.warn(f"failed to parse line: {e}")
                continue
            all_items.append(item)

    logger.info(f"loaded {len(all_items)} queries")

    # 2. 遍历 query 并计算长度
    for item in tqdm(all_items, desc="Processing queries"):
        if "candidates" in item.keys():
            candidates =  [ {"text": cand['text'], "token_confidence": cand['token_confidence']} for cand in item["candidates"]]
        elif "conversations" in item.keys():
            candidates = [ {"text": item["conversations"][1]["content"], "token_confidence": item["token_confidence"]} ]

        for cand in candidates[:cfg.num_candidates]:
            ans = extract_labels_from_text(cand['text'])
            score = compute_score(item, ans)  # 0-4
            
            is_correct = (score == 4) if not USE_SCORE_THRESHOLD_FOR_CORRECT else (score >= cfg.score_threshold)

            token_conf = cand.get("token_confidence", [])
            token_length = len(token_conf) 

            # 记录全量数据
            if is_correct:
                lengths_correct.append(token_length)
            else:
                lengths_incorrect.append(token_length)
                
            # 记录子集数据
            if token_length <= 200:
                if is_correct:
                    lengths_correct_zoomed.append(token_length)
                else:
                    lengths_incorrect_zoomed.append(token_length)

    logger.success("Finished computing lengths.")

    # 3. 绘图 (两个子图)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # 创建一个 1 行 2 列的子图布局
    bins = 50 
    
    # --- 左侧子图: 完整长度分布 ---
    # 使用全量数据
    axes[0].hist(lengths_correct, bins=bins, alpha=0.6, label=f'Correct (N={len(lengths_correct)})', density=False, color='skyblue')
    axes[0].hist(lengths_incorrect, bins=bins, alpha=0.6, label=f'Incorrect (N={len(lengths_incorrect)})', density=False, color='salmon')
    axes[0].set_xlabel('Token Length') 
    axes[0].set_ylabel('Count')
    axes[0].set_title('Token Length Distribution: Full Range')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.5, linestyle='--')

    # --- 右侧子图: 长度200以内的细节图（重新统计分箱） ---
    # 使用子集数据，并设置 bin 范围以优化分辨率
    axes[1].hist(lengths_correct_zoomed, bins=bins, range=(0, 200), alpha=0.6, label=f'Correct (N={len(lengths_correct_zoomed)})', density=False, color='skyblue')
    axes[1].hist(lengths_incorrect_zoomed, bins=bins, range=(0, 200), alpha=0.6, label=f'Incorrect (N={len(lengths_incorrect_zoomed)})', density=False, color='salmon')
    axes[1].set_xlabel('Token Length') 
    axes[1].set_ylabel('Count')
    axes[1].set_title('Token Length Distribution (0-200, Resampled)')
    axes[1].set_xlim(0, 200) # 确保 X 轴显示范围准确
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.5, linestyle='--')

    plt.tight_layout()
    # 保存图片
    out_png = f"length_distribution.png"
    plt.savefig(out_png, dpi=200)
    logger.success(f"Saved figure to {out_png}")
    plt.show()

    # 4. 输出统计量
    def stats(arr):
        if not arr:
            return {"n":0, "mean":None, "median":None}
        return {"n": len(arr), "mean": float(np.mean(arr)), "median": float(np.median(arr))}
        
    def stats_zoomed(arr):
        # 针对小于200的子集进行统计
        filtered_arr = [l for l in arr if l <= 200]
        return stats(filtered_arr)

    print("\nSummary statistics (Full Range Token Length):")
    print("Correct:", stats(lengths_correct))
    print("Incorrect:", stats(lengths_incorrect))
    
    print("\nSummary statistics (0-200 Token Length):")
    print("Correct:", stats_zoomed(lengths_correct))
    print("Incorrect:", stats_zoomed(lengths_incorrect))


def main():
    cfg = FilterConfig()
    logger = Logger()
    logger.info("Using config:")
    for k, v in vars(cfg).items():
        logger.info(f"  {k}: {v}")

    load_and_compute(cfg)

if __name__ == "__main__":
    main()
