# plot_confidence_distributions.py
import os
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal
from tqdm import tqdm
from collections import Counter, defaultdict
from tabulate import tabulate
import numpy as np
import re
import matplotlib.pyplot as plt

# ================== ⚙️ 配置类 ==================
@dataclass
class FilterConfig:
    # 输入 / 输出（只读取 input_file）
    input_file: str = "/workspace/mnt/lxb_work/zez_work/GuardRail/temp/synth_data_benign-inference.jsonl"

    # 评分与置信度计算相关
    score_threshold: Literal[1, 2, 3, 4] = 3
    confidence_calc_method: Literal[
        "average_trace_confidence", 
        "bottom_percent_group_confidence",
        "group_confidence",
        "lowest_group_confidence",
        "tail_confidence",
        "tail_confidence_by_percent"] = "bottom_percent_group_confidence"

    # 用于部分置信度计算的参数
    window_size: int = 1024
    tail_size: int = 1024
    tail_percent: float = 0.1

    num_candidates: int = 1         # 使用多少候选

# 如果为 True，则把 score >= score_threshold 的 candidate 也视为“正确”；
# 如果为 False，则仅把 compute_score(...) == 4 的 candidate 视为正确。
# （设为 True 可以和原脚本的筛选逻辑一致）
USE_SCORE_THRESHOLD_FOR_CORRECT = False

# ================ Logger 简单版（用于提示） ================
class Logger():
    def info(self, msg): print(f"[INFO] {msg}")
    def warn(self, msg): print(f"[WARN] {msg}")
    def success(self, msg): print(f"[OK] {msg}")
    def stat(self, msg): print(f"[STAT] {msg}")

# ================== ConfidenceCalculator（保持原逻辑） ==================
class ConfidenceCalculator:
    def __init__(self, token_confidences: List[float]):
        self.conf_list = np.array(token_confidences, dtype=float)
        self.n_tokens = len(token_confidences)

    def compute_average_trace_confidence(self) -> float:
        return float(np.mean(self.conf_list)) if self.n_tokens>0 else 0.0

    def compute_group_confidence_positional(self, window_size: int = 2048) -> np.ndarray:
        conf = self.conf_list
        n = self.n_tokens
        w = window_size
        if n == 0:
            return np.array([])
        prefix = np.zeros(n + 1)
        prefix[1:] = np.cumsum(conf)
        left = np.arange(n) - w + 1
        left = np.maximum(left, 0)
        lengths = np.arange(n) - left + 1
        window_sums = prefix[np.arange(1, n + 1)] - prefix[left]
        group_conf = window_sums / lengths
        return group_conf

    def compute_group_confidence(self, window_size: int = 2048) -> float:
        arr = self.compute_group_confidence_positional(window_size)
        return float(np.mean(arr)) if arr.size>0 else self.compute_average_trace_confidence()

    def compute_bottom_percent_group_confidence(self, window_size: int = 2048, bottom_percent: float = 10.0) -> float:
        group_conf = self.compute_group_confidence_positional(window_size)
        if group_conf.size == 0:
            return self.compute_average_trace_confidence()
        # 只考虑满窗口后的部分（与原脚本一致）
        if window_size-1 < len(group_conf):
            valid = group_conf[window_size-1:]
        else:
            valid = group_conf
        if valid.size == 0:
            return float(np.mean(group_conf))
        k = max(1, int(len(valid) * bottom_percent / 100.0))
        bottom_k = np.partition(valid, k-1)[:k]
        return float(np.mean(bottom_k))

    def compute_lowest_group_confidence(self, window_size: int = 2048) -> float:
        group_conf = self.compute_group_confidence_positional(window_size)
        if group_conf.size == 0:
            return float(np.min(self.conf_list)) if self.n_tokens>0 else 0.0
        if window_size-1 < len(group_conf):
            valid = group_conf[window_size-1:]
        else:
            valid = group_conf
        return float(np.min(valid)) if valid.size>0 else float(np.min(group_conf))

    def compute_tail_confidence(self, tail_size: int = 2048) -> float:
        if self.n_tokens == 0: return 0.0
        tail_tokens = self.conf_list[-tail_size:]
        return float(np.mean(tail_tokens))

    def compute_tail_confidence_by_percent(self, tail_percent: float = 0.1) -> float:
        if self.n_tokens == 0: return 0.0
        tail_size = max(1, int(self.n_tokens * tail_percent))
        tail_tokens = self.conf_list[-tail_size:]
        return float(np.mean(tail_tokens))

    def compute_confidence_by_method(self, method: str, cfg: FilterConfig):
        if method == "average_trace_confidence":
            return self.compute_average_trace_confidence()
        if method == "bottom_percent_group_confidence":
            return self.compute_bottom_percent_group_confidence(window_size=cfg.window_size, bottom_percent=10.0)
        if method == "group_confidence":
            return self.compute_group_confidence(window_size=cfg.window_size)
        if method == "lowest_group_confidence":
            return self.compute_lowest_group_confidence(window_size=cfg.window_size)
        if method == "tail_confidence":
            return self.compute_tail_confidence(tail_size=cfg.tail_size)
        if method == "tail_confidence_by_percent":
            return self.compute_tail_confidence_by_percent(tail_percent=cfg.tail_percent)
        raise ValueError("unsupported confidence_calc_method")

# ================== 提取答案与评分函数（保留原逻辑） ==================
def extract_labels_from_text(text: str, raw_answer: bool = False):
    think_pattern = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
    think_match = think_pattern.search(text)
    think_content = think_match.group(1).strip() if think_match else None
    outside_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    answer_pattern = re.compile(
        r"Request\s*[:\-]?\s*(harmful|unharmful|none|null)\b.*?"
        r"Completion\s*[:\-]?\s*(refusal|compliance|none|null)\b.*?"
        r"Response\s*[:\-]?\s*(harmful|unharmful|none|null)\b",
        flags=re.IGNORECASE | re.DOTALL
    )
    answer_match = answer_pattern.search(outside_text)
    def normalize(value: str) -> str:
        if value is None: return None
        value = value.lower().strip()
        if value in ["harmful"]: return "harmful"
        if value in ["unharmful", "harmless", "not harmful"]: return "unharmful"
        if value in ["refusal", "compliance"]: return value
        if value in ["none", "null", "empty"]: return None
        return value
    if answer_match:
        prompt_harm = normalize(answer_match.group(1))
        response_refusal = normalize(answer_match.group(2))
        response_harm = normalize(answer_match.group(3))
        answer_dict = {
            "prompt_harm": prompt_harm,
            "response_harm": response_harm,
            "response_refusal": response_refusal
        }
    else:
        answer_dict = None
    if raw_answer:
        return outside_text, answer_dict, think_content
    return outside_text, answer_dict

def compute_score(item, answer_dict):
    prompt_harm_label = item.get("prompt_harm_label")
    response_refusal_label = item.get("response_refusal_label")
    response_harm_label = item.get("response_harm_label")
    score = 0
    if answer_dict is None:
        return score
    score = 1
    if answer_dict["prompt_harm"] != prompt_harm_label: 
        return score
    score = 2
    if answer_dict["response_harm"] != response_harm_label:
        return score
    score = 3
    if answer_dict["response_refusal"] != response_refusal_label:
        return score
    score = 4
    return score

# ================== 主处理与绘图 ==================
def load_and_compute(cfg: FilterConfig):
    logger = Logger()
    if not os.path.exists(cfg.input_file):
        logger.warn(f"input file not found: {cfg.input_file}")
        return

    # 准备容器
    confidences_correct = []  # 每个 candidate（正确）的置信度
    confidences_incorrect = []  # 每个 candidate（错误）的置信度

    # 对 query 分类统计（按正确candidate数量分类）
    # types: 'boundary', 'outlier', 'in_distribution'
    type_confidences = {
        "boundary": [], 
        "outlier": [], 
        "in_distribution": []
    }

    # 对 query 分类统计（按正确任务类别）
    # types: 'user', 'agent'
    task_type_confidences = {
        "user": [],
        "agent": []
    }

    # 读取所有数据并计算每个 candidate 的置信度与正确/错误标签
    all_items = []
    with open(cfg.input_file, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Reading input"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as e:
                logger.warn(f"failed to parse line: {e}")
                continue
            all_items.append(item)

    logger.info(f"loaded {len(all_items)} queries")

    # 遍历 query
    for item in tqdm(all_items, desc="Processing queries"):
        qid = item.get("id", None)
        candidates = item.get("candidates", [])
        response = item.get("response")
        # 首先对该 query 的所有 candidate 计算是否“正确”（score）并得到置信度
        per_query_confidences = []
        per_query_correct_flags = []

        for cand in candidates[:FilterConfig.num_candidates]:
            cand_text = cand.get("text", "")
            _, ans = extract_labels_from_text(cand_text)
            score = compute_score(item, ans)  # 0-4
            # 是否视为正确（两种策略）
            if USE_SCORE_THRESHOLD_FOR_CORRECT:
                is_correct = (score >= cfg.score_threshold)
            else:
                is_correct = (score == 4)

            token_conf = cand.get("token_confidence", [])
            conf_calc = ConfidenceCalculator(token_confidences=token_conf)
            conf_val = conf_calc.compute_confidence_by_method(cfg.confidence_calc_method, cfg)

            # 记录
            per_query_confidences.append(conf_val)
            per_query_correct_flags.append(bool(is_correct))

            # 全局分类（正确 vs 错误）
            if is_correct:
                confidences_correct.append(conf_val)
            else:
                confidences_incorrect.append(conf_val)

        # 根据该 query 的正确 candidate 数量来判定类型
        correct_count = sum(1 for f in per_query_correct_flags if f)
        # 假设：correct_count ∈ {0..8}，按你给的样例映射：
        if correct_count in {3,4,5}:
            qtype = "boundary"
        elif correct_count in {0,1,2}:
            qtype = "outlier"
        elif correct_count in {6,7,8}:
            qtype = "in_distribution"
        else:
            # 如果出现其他数量（超出 0-8），按最接近的分组归入 boundary（保守处理）
            if correct_count < 3:
                qtype = "outlier"
            elif correct_count > 5:
                qtype = "in_distribution"
            else:
                qtype = "boundary"

        # 把该 query 下所有 candidate 的置信度都计入该 query type 的置信度集合（便于画分布）
        type_confidences[qtype].extend(per_query_confidences)

        # 把 query 按任务类别分类
        if response is None:
            task_type_confidences["user"].extend(per_query_confidences)
        else:
            task_type_confidences["agent"].extend(per_query_confidences)

    logger.success("Finished computing confidences.")

    # ---------- 绘图：3-in-1 (三个子图) ----------
    bins = 50
    plt.figure(figsize=(15, 4))

    # 子图 1: Correct vs Incorrect
    ax1 = plt.subplot(1, 3, 1)
    ax1.hist(confidences_correct, bins=bins, alpha=0.6, label='Correct', density=False)
    ax1.hist(confidences_incorrect, bins=bins, alpha=0.6, label='Incorrect', density=False)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title('Confidence distribution: Correct vs Incorrect')
    ax1.legend()
    ax1.grid(alpha=0.25)

    # 子图 2: Boundary / Outlier / In-Distribution
    ax2 = plt.subplot(1, 3, 2)
    ax2.hist(type_confidences["boundary"], bins=bins, alpha=0.5, label='Boundary', density=False)
    ax2.hist(type_confidences["outlier"], bins=bins, alpha=0.5, label='Outlier', density=False)
    ax2.hist(type_confidences["in_distribution"], bins=bins, alpha=0.5, label='In-Distribution', density=False)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence distribution by query type')
    ax2.legend()
    ax2.grid(alpha=0.25)

    # 子图 3: Task type (user vs agent)
    ax3 = plt.subplot(1, 3, 3)
    ax3.hist(task_type_confidences["user"], bins=bins, alpha=0.6, label='User', density=False)
    ax3.hist(task_type_confidences["agent"], bins=bins, alpha=0.6, label='Agent', density=False)
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Count')
    ax3.set_title('Confidence distribution by task type')
    ax3.legend()
    ax3.grid(alpha=0.25)

    plt.tight_layout()

    out_png = f"{FilterConfig.confidence_calc_method}_distributions_by_task_benign.png"
    plt.savefig(out_png, dpi=200)
    logger.success(f"Saved figure to {out_png}")
    plt.show()

    # ---------- 统计量输出 ----------
    def stats(arr):
        if not arr:
            return {"n":0, "mean":None, "median":None}
        return {"n": len(arr), "mean": float(np.mean(arr)), "median": float(np.median(arr))}

    print("\nSummary statistics:")
    print("Correct:", stats(confidences_correct))
    print("Incorrect:", stats(confidences_incorrect))
    for t in ["boundary","outlier","in_distribution"]:
        print(f"{t}:", stats(type_confidences[t]))

    # 任务类别（user/agent）统计
    print("\nTask-type statistics:")
    print("User:", stats(task_type_confidences["user"]))
    print("Agent:", stats(task_type_confidences["agent"]))

def main():
    cfg = FilterConfig()
    logger = Logger()
    logger.info("Using config:")
    for k, v in vars(cfg).items():
        logger.info(f"  {k}: {v}")

    load_and_compute(cfg)

if __name__ == "__main__":
    main()
