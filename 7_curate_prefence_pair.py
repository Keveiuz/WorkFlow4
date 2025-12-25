#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
构建 DPO 偏好对脚本（单文件）
输出格式（每行 JSON）：
{
  "id": "123",
  "prompt": "...",
  "chosen": "chosen 文本",
  "rejected": "rejected 文本",
  "chosen_conf": 0.9123,
  "rejected_conf": 0.4212,
  "confidence_diff": 0.4911
}
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal, Any
from tqdm import tqdm
from collections import Counter
from tabulate import tabulate
import numpy as np
import re
import sys

# ================== ⚙️ 配置类 ==================
@dataclass
class FilterConfig:
    # 输入 / 输出（请根据你的环境修改）
    input_file: str = "/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/guardreasoner-reinference.jsonl"
    output_file: str = "/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/guardreasoner-preference.jsonl"

    # 原始评分阈值（用于挑选 chosen / rejected）
    format_score_threshold: Literal[1, 2, 3, 4] = 1     # 格式正确分数阈值
    wrong_score_threshold: Literal[1, 2, 3, 4] = 2      # 错误回答分数阈值
    correct_score_threshold: Literal[1, 2, 3, 4] = 4    # 正确回答分数阈值

    # 置信度计算方法
    confidence_calc_method: Literal[
        "average_trace_confidence",
        "bottom_percent_group_confidence",
        "group_confidence",
        "lowest_group_confidence",
        "tail_confidence",
        "tail_confidence_by_percent"] = "bottom_percent_group_confidence"

    # 全局筛选（optional，用于在 pair 构建后对 score 排序进行全局截断）
    confidence_threshold: Optional[float] = None        # 作用为：仅保留 pair.score >= threshold（score 由 dpo 模式定义）
    select_ratio: Optional[float] = None               # 全局选择 top N% 的 queries（基于 pair.score）
    select_n: Optional[int] = None                     # 全局选择 totail_confidence_by_percenp N 的 queries（基于 pair.score）

    # DPO 策略：closest / high_high / low_low / high_low / low_high
    dpo_mode: Literal[
        "closest", 
        "high_high", 
        "low_low", 
        "high_low", 
        "low_high"] = "low_high"

    # closest 模式参数
    closest_sort: Literal["desc", "asc"] = None         # 差值排序方向（desc: 从大到小）
    closest_threshold: Optional[float] = None             # 差值阈值（绝对值）——若提供则仅保留 closest >= 阈值 的 pair
    closest_top_n: Optional[int] = None                   # 若提供则仅取排序后的前 N 个 pair

    # 高高/低低/高低/低高 模式参数
    threshold_mode: bool = False                       # False: query 内自动选最高/最低；True: 使用阈值筛选
    chosen_conf_threshold: Optional[float] = None      # threshold_mode 为 True 时生效（chosen 的阈值）
    rejected_conf_threshold: Optional[float] = None    # threshold_mode 为 True 时生效（rejected 的阈值）
    length_control: Optional[float] = len("<think>\nAnswers:  \nRequest: unharmful  \nCompletion: compliance  \nResponse: unharmful\n</think>\n\n Request: unharmful \nCompletion: compliance \nResponse: unharmful") + 5

    # 使用不同的置信度计算方法时需要的窗口 / 百分比参数
    window_size: int = 1024
    bottom_percent: float = 0.1
    tail_size: int = 1024
    tail_percent: float = 0.1

    # 是否在最终输出中去掉 token_confidence 原始数组（默认不保留）
    remove_token_confidence: bool = True

    # 其他：输出调试信息
    verbose: bool = True

# ================== 🔧 工具类 ==================
class Logger():
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.__GREEN__ = "\033[92m"; self.__YELLOW__ = "\033[93m"; self.__RED__ = "\033[91m"; self.__BLUE__ = "\033[94m"; self.__MAGENTA__ = "\033[95m"; self.__CYAN__ = "\033[96m"; self.__GRAY__ = "\033[90m"; self.__CRITICAL__ = "\033[41;97m"; self.__RESET__ = "\033[0m"
    def success(self, msg): 
        if self.verbose: print(f"{self.__GREEN__}[SUCCESS]{self.__RESET__} {msg}")
    def warn(self, msg): 
        if self.verbose: print(f"{self.__YELLOW__}[WARNING]{self.__RESET__} {msg}")
    def error(self, msg): 
        if self.verbose: print(f"{self.__RED__}[ERROR]{self.__RESET__} {msg}", file=sys.stderr)
    def debug(self, msg): 
        if self.verbose: print(f"{self.__BLUE__}[DEBUG]{self.__RESET__} {msg}")
    def stat(self, msg): 
        if self.verbose: print(f"{self.__MAGENTA__}[STAT]{self.__RESET__} {msg}")
    def info(self, msg): 
        if self.verbose: print(f"{self.__CYAN__}[INFO]{self.__RESET__} {msg}")
    def note(self, msg):
        if self.verbose: print(f"{self.__GRAY__}[NOTE]{self.__RESET__} {msg}")
    def critical(self, msg): 
        if self.verbose: print(f"{self.__CRITICAL__}[CRITICAL]{self.__RESET__} {msg}")

class Table():
    @staticmethod
    def draw_table(user_safe: int, user_unsafe: int, agent_safe: int, agent_unsafe: int):
        user_total = user_safe + user_unsafe
        agent_total = agent_safe + agent_unsafe
        safe_total = user_safe + agent_safe
        unsafe_total = user_unsafe + agent_unsafe
        grand_total = user_total + agent_total

        table_data = [
            ["user", user_safe, user_unsafe, user_total],
            ["agent", agent_safe, agent_unsafe, agent_total],
            ["", safe_total, unsafe_total, ""]
        ]
        colalign = ("center", "center", "center", "center")
        table_str = tabulate(table_data, headers=["", "safe", "unsafe", ""], tablefmt="grid", colalign=colalign)
        print(table_str)
        width = max(len(line) for line in table_str.split("\n"))
        total_str = f" total: {grand_total} "
        padding = width - 2 - len(total_str)
        left_pad = padding // 2
        right_pad = padding - left_pad
        print("|" + total_str + " " * (right_pad + left_pad) + "|")
        print("-" * width)

# ================== 🧮 置信度计算类 ==================
class ConfidenceCalculator:
    """
    假设输入是每个token位置的confidence值列表
    """
    def __init__(self, token_confidences: List[float]):
        self.conf_list = np.array(token_confidences, dtype=float) if token_confidences is not None else np.array([])
        self.n_tokens = int(len(self.conf_list))

    def compute_average_trace_confidence(self) -> float:
        if self.n_tokens == 0: return 0.0
        return float(np.mean(self.conf_list))

    def compute_group_confidence_positional(self, window_size: int = 2048) -> np.ndarray:
        if self.n_tokens == 0:
            return np.array([])
        conf = self.conf_list
        n = self.n_tokens
        w = max(1, window_size)
        prefix = np.zeros(n + 1, dtype=float)
        prefix[1:] = np.cumsum(conf)
        left = np.arange(n) - w + 1
        left = np.maximum(left, 0)
        lengths = np.arange(n) - left + 1
        window_sums = prefix[np.arange(1, n + 1)] - prefix[left]
        group_conf = window_sums / lengths
        return group_conf

    def compute_group_confidence(self, window_size: int = 2048) -> float:
        arr = self.compute_group_confidence_positional(window_size=window_size)
        return float(np.mean(arr)) if arr.size > 0 else self.compute_average_trace_confidence()

    def compute_bottom_percent_group_confidence(self, window_size: int = 2048, bottom_percent: float = 0.1) -> float:
        arr = self.compute_group_confidence_positional(window_size=window_size)
        if arr.size == 0:
            return self.compute_average_trace_confidence()
        valid = arr[max(0, window_size-1):]
        if valid.size == 0:
            return float(np.mean(arr))
        k = max(1, int(len(valid) * bottom_percent))
        bottom = np.partition(valid, k-1)[:k]
        return float(np.mean(bottom))

    def compute_lowest_group_confidence(self, window_size: int = 2048) -> float:
        arr = self.compute_group_confidence_positional(window_size=window_size)
        valid = arr[max(0, window_size-1):]
        if valid.size == 0:
            return float(np.min(self.conf_list)) if self.n_tokens>0 else 0.0
        return float(np.min(valid))

    def compute_tail_confidence(self, tail_size: int = 2048) -> float:
        if self.n_tokens == 0: return 0.0
        tail = self.conf_list[-tail_size:]
        return float(np.mean(tail))

    def compute_tail_confidence_by_percent(self, tail_percent: float = 0.1) -> float:
        if self.n_tokens == 0: return 0.0
        tail_size = max(1, int(self.n_tokens * tail_percent))
        tail = self.conf_list[-tail_size:]
        return float(np.mean(tail))

# ================== 文本解析 & 得分函数 ==================
REASONING_TEMPLATE = """System Prompt:
You are a classifier for analyzing interactions between humans and LLMs. 

Given a request from a human user and a response from an LLM assistant, finish three tasks.

Task 1: determine whether the human user's request is harmful or unharmful.
Task 2: determine whether the AI assistant's response is a refusal or compliance.
Task 3: determine whether the AI assistant's response is harmful or unharmful.

You must think step by step. Keep consistency between the reasoning conclusion and final result.

User Prompt:
Human user:{prompt}
AI assistant:{response}

Expected Output Format:
Answers: 
Request: [harmful/unharmful]
Completion: [refusal/compliance/null]
Response: [harmful/unharmful/null]"""

def extract_labels_from_text(text: str, raw_answer: bool = False):
    """
    从 candidate.text 中提取 <think> ... </think> 与最终三元答案（Request/Completion/Response）
    返回 outside_text, answer_dict 或 (outside_text, answer_dict, think_content) if raw_answer True
    """
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
    def normalize(value: str) -> Optional[str]:
        if value is None: return None
        v = value.lower().strip()
        if v in ["harmful"]:
            return "harmful"
        if v in ["unharmful", "harmless", "not harmful"]:
            return "unharmful"
        if v in ["refusal", "compliance"]:
            return v
        if v in ["none", "null", "empty"]:
            return None
        return v

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

def compute_score(item: dict, answer_dict: Optional[dict]) -> int:
    """
    返回 score（0-4）
    """
    prompt_harm_label = item.get("prompt_harm_label")
    response_refusal_label = item.get("response_refusal_label")
    response_harm_label = item.get("response_harm_label")
    score = 0 
    if answer_dict is None:
        return score
    score = 1
    if answer_dict.get("prompt_harm") != prompt_harm_label:
        return score
    score = 2
    if answer_dict.get("response_harm") != response_harm_label:
        return score
    score = 3
    if answer_dict.get("response_refusal") != response_refusal_label:
        return score
    score = 4
    return score

# ================== DPO 配对选择器 ==================
def select_preference_pair_for_query(qid: Any, chosen_list: List[dict], rejected_list: List[dict], cfg: FilterConfig, logger: Logger):
    """
    从单个 query 的 chosen_list 与 rejected_list 中构造一个 pair（或返回 None）
    每个 chosen_list/rejected_list 的元素结构: {"candidate": cand_obj, "confidence": float}
    返回 dict:
        {
           "qid": qid,
           "chosen": chosen_entry,
           "rejected": rejected_entry,
           "chosen_confidence": float,
           "rejected_confidence": float,
           "confidence_diff": float,
           "score": float   # 用于全局排序/筛选（通常等于 confidence_diff 的绝对值或按策略定义）
        }
    """
    if not chosen_list or not rejected_list:
        return None

    # 防止传入的 lists 被修改
    chosen_sorted = sorted(chosen_list, key=lambda x: x["confidence"])
    rejected_sorted = sorted(rejected_list, key=lambda x: x["confidence"])

    mode = cfg.dpo_mode

    # Helper: safe access to thresholds
    chosen_thr = cfg.chosen_conf_threshold
    rejected_thr = cfg.rejected_conf_threshold

    # MODE: closest
    if mode == "closest":
        min_diff = float("inf")
        best_pair = None

        # 遍历所有 chosen 与 rejected 组合，找出 confidence 差值最小的
        for chosen_entry in chosen_sorted:
            chosen_conf = float(chosen_entry["confidence"])
            for rejected_entry in rejected_sorted:
                rejected_conf = float(rejected_entry["confidence"])
                diff = abs(chosen_conf - rejected_conf)
                if diff < min_diff:
                    min_diff = diff
                    best_pair = {
                        "chosen": chosen_entry,
                        "rejected": rejected_entry,
                        "chosen_confidence": chosen_conf,
                        "rejected_confidence": rejected_conf,
                        "confidence_diff": diff,
                        "score": diff  # 用于排序的 score（绝对差）
                    }
        if best_pair is not None:
            best_pair["qid"] = qid
            return best_pair

    # MODE: high_high
    if mode == "high_high":
        if not cfg.threshold_mode:
            # 在同一 query 中直接取 chosen 最大, rejected 最大
            chosen_entry = chosen_sorted[-1]
            rejected_entry = rejected_sorted[-1]
            chosen_conf = float(chosen_entry["confidence"])
            rejected_conf = float(rejected_entry["confidence"])
            return {
                "qid": qid,
                "chosen": chosen_entry,
                "rejected": rejected_entry,
                "chosen_confidence": chosen_conf,
                "rejected_confidence": rejected_conf,
                "confidence_diff": chosen_conf - rejected_conf,
                "score": (chosen_conf + rejected_conf) / 2.0
            }
        else:
            # threshold_mode: 需要 chosen_conf >= chosen_thr 且 rejected_conf >= rejected_thr
            if chosen_thr is None or rejected_thr is None:
                logger.warn(f"threshold_mode is True but chosen/rejected thresholds not set for qid={qid}. Skipping.")
                return None
            cand_c = [c for c in chosen_list if c["confidence"] >= chosen_thr]
            cand_r = [r for r in rejected_list if r["confidence"] >= rejected_thr]
            if not cand_c or not cand_r:
                return None
            chosen_entry = max(cand_c, key=lambda x: x["confidence"])
            rejected_entry = max(cand_r, key=lambda x: x["confidence"])
            chosen_conf = float(chosen_entry["confidence"])
            rejected_conf = float(rejected_entry["confidence"])
            return {
                "qid": qid,
                "chosen": chosen_entry,
                "rejected": rejected_entry,
                "chosen_confidence": chosen_conf,
                "rejected_confidence": rejected_conf,
                "confidence_diff": chosen_conf - rejected_conf,
                "score": (chosen_conf + rejected_conf) / 2.0
            }

    # MODE: low_low
    if mode == "low_low":
        if not cfg.threshold_mode:
            chosen_entry = chosen_sorted[0]
            rejected_entry = rejected_sorted[0]
            chosen_conf = float(chosen_entry["confidence"])
            rejected_conf = float(rejected_entry["confidence"])
            return {
                "qid": qid,
                "chosen": chosen_entry,
                "rejected": rejected_entry,
                "chosen_confidence": chosen_conf,
                "rejected_confidence": rejected_conf,
                "confidence_diff": chosen_conf - rejected_conf,
                "score": (chosen_conf + rejected_conf) / 2.0
            }
        else:
            if chosen_thr is None or rejected_thr is None:
                logger.warn(f"threshold_mode is True but chosen/rejected thresholds not set for qid={qid}. Skipping.")
                return None
            cand_c = [c for c in chosen_list if c["confidence"] <= chosen_thr]
            cand_r = [r for r in rejected_list if r["confidence"] <= rejected_thr]
            if not cand_c or not cand_r:
                return None
            chosen_entry = min(cand_c, key=lambda x: x["confidence"])
            rejected_entry = min(cand_r, key=lambda x: x["confidence"])
            chosen_conf = float(chosen_entry["confidence"])
            rejected_conf = float(rejected_entry["confidence"])
            return {
                "qid": qid,
                "chosen": chosen_entry,
                "rejected": rejected_entry,
                "chosen_confidence": chosen_conf,
                "rejected_confidence": rejected_conf,
                "confidence_diff": chosen_conf - rejected_conf,
                "score": (chosen_conf + rejected_conf) / 2.0
            }

    # MODE: high_low
    if mode == "high_low":
        if not cfg.threshold_mode:
            chosen_entry = chosen_sorted[-1]
            rejected_entry = rejected_sorted[0]
            chosen_conf = float(chosen_entry["confidence"])
            rejected_conf = float(rejected_entry["confidence"])
            return {
                "qid": qid,
                "chosen": chosen_entry,
                "rejected": rejected_entry,
                "chosen_confidence": chosen_conf,
                "rejected_confidence": rejected_conf,
                "confidence_diff": chosen_conf - rejected_conf,
                "score": chosen_conf - rejected_conf
            }
        else:
            if chosen_thr is None or rejected_thr is None:
                logger.warn(f"threshold_mode is True but chosen/rejected thresholds not set for qid={qid}. Skipping.")
                return None
            cand_c = [c for c in chosen_list if c["confidence"] >= chosen_thr]
            cand_r = [r for r in rejected_list if r["confidence"] <= rejected_thr]
            if not cand_c or not cand_r:
                return None
            chosen_entry = max(cand_c, key=lambda x: x["confidence"])
            rejected_entry = min(cand_r, key=lambda x: x["confidence"])
            chosen_conf = float(chosen_entry["confidence"])
            rejected_conf = float(rejected_entry["confidence"])
            return {
                "qid": qid,
                "chosen": chosen_entry,
                "rejected": rejected_entry,
                "chosen_confidence": chosen_conf,
                "rejected_confidence": rejected_conf,
                "confidence_diff": chosen_conf - rejected_conf,
                "score": chosen_conf - rejected_conf
            }

    # MODE: low_high
    if mode == "low_high":
        if not cfg.threshold_mode:
            chosen_entry = chosen_sorted[0]
            rejected_entry = rejected_sorted[-1]
            chosen_conf = float(chosen_entry["confidence"])
            rejected_conf = float(rejected_entry["confidence"])
            return {
                "qid": qid,
                "chosen": chosen_entry,
                "rejected": rejected_entry,
                "chosen_confidence": chosen_conf,
                "rejected_confidence": rejected_conf,
                "confidence_diff": chosen_conf - rejected_conf,
                "score": rejected_conf - chosen_conf
            }
        else:
            if chosen_thr is None or rejected_thr is None:
                logger.warn(f"threshold_mode is True but chosen/rejected thresholds not set for qid={qid}. Skipping.")
                return None
            cand_c = [c for c in chosen_list if c["confidence"] <= chosen_thr]
            cand_r = [r for r in rejected_list if r["confidence"] >= rejected_thr]
            if not cand_c or not cand_r:
                return None
            chosen_entry = min(cand_c, key=lambda x: x["confidence"])
            rejected_entry = max(cand_r, key=lambda x: x["confidence"])
            chosen_conf = float(chosen_entry["confidence"])
            rejected_conf = float(rejected_entry["confidence"])
            return {
                "qid": qid,
                "chosen": chosen_entry,
                "rejected": rejected_entry,
                "chosen_confidence": chosen_conf,
                "rejected_confidence": rejected_conf,
                "confidence_diff": chosen_conf - rejected_conf,
                "score": rejected_conf - chosen_conf
            }

    return None

# ================== 主流程 ==================
def filter_candidates(cfg: FilterConfig):
    logger = Logger(verbose=cfg.verbose)

    logger.info(f"Loading data from {cfg.input_file}")
    all_data = []
    try:
        total_lines = sum(1 for _ in open(cfg.input_file, "r", encoding="utf-8"))
        with open(cfg.input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc="Reading input file", unit="lines"):
                line = line.strip()
                if not line:
                    continue
                try:
                    all_data.append(json.loads(line))
                except Exception as e:
                    logger.warn(f"failed to parse line: {e}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {cfg.input_file}")
        return
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return

    logger.info(f"Total queries loaded: {len(all_data)}")
    query_map = { item.get("id"): item for item in all_data }

    # 2) 计算 candidate 的 score（格式准确度）并计算每个 candidate 的 confidence
    valid_queries: Dict[Any, Dict[str, List[dict]]] = {}  # qid -> {"chosen": [...], "rejected":[...]}
    all_conf_values = []

    for item in tqdm(all_data, desc="Validating candidates", unit="queries"):
        qid = item.get("id")
        if qid is None:
            continue
        valid_queries.setdefault(qid, {"chosen": [], "rejected": []})
        for cand in item.get("candidates", []):
            resp_text = cand.get("text", "")

            # 先筛选长度合格的 candidate（杜绝没有推理只有答案的情况）
            if FilterConfig.length_control is not None and len(resp_text) < FilterConfig.length_control:
                continue

            _, ans = extract_labels_from_text(resp_text)
            score = compute_score(item, ans)

            # 先筛选格式正确的 candidate（score > format threshold）
            if score <= cfg.format_score_threshold:
                continue

            # 计算置信度
            token_conf = cand.get("token_confidence", []) or []
            conf_calc = ConfidenceCalculator(token_confidences=token_conf)
            try:
                if cfg.confidence_calc_method == "average_trace_confidence":
                    conf = conf_calc.compute_average_trace_confidence()
                elif cfg.confidence_calc_method == "bottom_percent_group_confidence":
                    conf = conf_calc.compute_bottom_percent_group_confidence(window_size=cfg.window_size, bottom_percent=cfg.bottom_percent)
                elif cfg.confidence_calc_method == "group_confidence":
                    conf = conf_calc.compute_group_confidence(window_size=cfg.window_size)
                elif cfg.confidence_calc_method == "lowest_group_confidence":
                    conf = conf_calc.compute_lowest_group_confidence(window_size=cfg.window_size)
                elif cfg.confidence_calc_method == "tail_confidence":
                    conf = conf_calc.compute_tail_confidence(tail_size=cfg.tail_size)
                elif cfg.confidence_calc_method == "tail_confidence_by_percent":
                    conf = conf_calc.compute_tail_confidence_by_percent(tail_percent=cfg.tail_percent)
                else:
                    raise TypeError("unsupported confidence calculate method")
            except Exception as e:
                logger.warn(f"Failed to compute confidence for qid={qid}: {e}")
                conf = 0.0

            conf_val = float(conf) if conf is not None else 0.0
            # 收集全局分布
            all_conf_values.append(conf_val)

            valid_entry = {"candidate": cand, "confidence": conf_val}

            # rejected: score in [wrong_score_threshold, correct_score_threshold)
            if cfg.wrong_score_threshold <= score < cfg.correct_score_threshold:
                valid_queries[qid]["rejected"].append(valid_entry)

            # chosen: score >= correct_score_threshold
            if score >= cfg.correct_score_threshold:
                valid_queries[qid]["chosen"].append(valid_entry)

    # 仅保留那些同时有 chosen 和 rejected 的 query
    qualified_qids = [qid for qid, lists in valid_queries.items() if lists["chosen"] and lists["rejected"]]
    logger.info(f"Total queries with at least one chosen & one rejected: {len(qualified_qids)}")
    all_correct = [qid for qid, lists in valid_queries.items() if len(lists["rejected"]) == 0 and len(lists["chosen"]) != 0]
    logger.info(f"All correct queries: {len(all_correct)}")
    all_wrong = [qid for qid, lists in valid_queries.items() if len(lists["chosen"]) == 0 and len(lists["rejected"]) != 0]
    logger.info(f"All wrong queries: {len(all_wrong)}")

    # ========== 在原始数据（valid_queries）中统计未约减的 chosen:rejected 比例 ==========
    ratio_counter = Counter()
    total_queries_for_ratio = 0

    for qid, lists in valid_queries.items():
        c = len(lists.get("chosen", []))
        r = len(lists.get("rejected", []))
        total_queries_for_ratio += 1

        ratio_str = f"{c}:{r}"
        # if ratio_str == "0:0":
        #     print(qid)
        ratio_counter[ratio_str] += 1

    logger.stat("======== Raw chosen:rejected ratio distribution (NO reduction) ========")
    if total_queries_for_ratio == 0:
        logger.stat("No queries available to compute ratio distribution.")
    else:
        # 按出现次数从大到小排序
        # 自定义排序：先按 n 降序，再按 m 降序
        sorted_ratios = sorted(
            ratio_counter.items(),
            key=lambda x: (int(x[0].split(":")[0]), int(x[0].split(":")[1])),
            reverse=True
        )

        for ratio, cnt in sorted_ratios:
            pct = cnt / total_queries_for_ratio * 100.0
            logger.stat(f"ratio {ratio} -> {cnt} queries ({pct:.2f}%)")
        logger.stat(f"total queries considered = {total_queries_for_ratio}")
    logger.stat("====================================================================")



    if len(qualified_qids) == 0:
        logger.info("No qualified (chosen + rejected) queries found. Exiting.")
        return

    # 如果用户设置了 threshold_mode = True 且没有提供 chosen/rejected 阈值，
    # 我们可以自动基于全局置信度分布设置默认阈值（25%/75% 分位），并提示用户
    if cfg.threshold_mode:
        if (cfg.chosen_conf_threshold is None) or (cfg.rejected_conf_threshold is None):
            if len(all_conf_values) > 0:
                arr = np.array(all_conf_values)
                default_high = float(np.percentile(arr, 75))
                default_low = float(np.percentile(arr, 25))
                # 如果某一阈值未设，则分别设置成 high/low
                if cfg.chosen_conf_threshold is None:
                    cfg.chosen_conf_threshold = default_high
                    logger.info(f"Auto-set chosen_conf_threshold = 75th percentile = {cfg.chosen_conf_threshold:.6f}")
                if cfg.rejected_conf_threshold is None:
                    cfg.rejected_conf_threshold = default_low
                    logger.info(f"Auto-set rejected_conf_threshold = 25th percentile = {cfg.rejected_conf_threshold:.6f}")
            else:
                logger.warn("threshold_mode requested but no global confidences available to compute defaults.")

    # 3) 对每个 qualified query，用 select_preference_pair_for_query 生成一个 pair（或 None）
    pairs = []
    for qid in tqdm(qualified_qids, desc="Constructing DPO pairs per query", unit="queries"):
        lists = valid_queries[qid]
        pair = select_preference_pair_for_query(qid, lists["chosen"], lists["rejected"], cfg, logger)
        if pair:
            pairs.append(pair)

    logger.info(f"Built {len(pairs)} initial pairs from qualified queries (one pair per query where possible).")

    if len(pairs) == 0:
        logger.info("No pairs constructed after selection rules. Exiting.")
        return

    # 4) 对 pairs 做 closest 模式下的额外筛选与排序（closest_top_n / closest_threshold / closest_sort）
    #    以及通用的 global confidence_threshold / select_ratio / select_n 筛选（基于 pair['score']）
    if cfg.dpo_mode == "closest":
        # 根据 closest_sort 决定排序方向
        reverse = True if cfg.closest_sort == "desc" else False
        pairs = sorted(pairs, key=lambda x: x.get("confidence_closest", 0.0), reverse=reverse)
        if cfg.closest_threshold is not None:
            pairs = [p for p in pairs if p.get("confidence_closest", 0.0) >= float(cfg.closest_threshold)]
            logger.info(f"After applying closest_threshold={cfg.closest_threshold}, pairs remain: {len(pairs)}")
        if cfg.closest_top_n is not None:
            pairs = pairs[:cfg.closest_top_n]
            logger.info(f"After applying closest_top_n={cfg.closest_top_n}, pairs remain: {len(pairs)}")

    # 通用过滤: confidence_threshold/select_ratio/select_n 基于 pair['score']
    if cfg.confidence_threshold is not None:
        pairs = [p for p in pairs if p.get("score", 0.0) >= float(cfg.confidence_threshold)]
        logger.info(f"After applying confidence_threshold={cfg.confidence_threshold}, pairs remain: {len(pairs)}")

    # 全局排序：默认以 score 降序（score 越高优先），若用户希望别的顺序可调整 cfg.select_ratio / select_n 与 dpo_mode 参数
    pairs = sorted(pairs, key=lambda x: x.get("score", 0.0), reverse=True)

    total_pairs = len(pairs)
    if cfg.select_ratio is not None:
        n_select = max(1, int(total_pairs * cfg.select_ratio))
        pairs = pairs[:n_select]
        logger.info(f"Applied select_ratio={cfg.select_ratio}, selecting top {n_select} pairs")
    elif cfg.select_n is not None:
        n_select = max(1, min(cfg.select_n, total_pairs))
        pairs = pairs[:n_select]
        logger.info(f"Applied select_n={cfg.select_n}, selecting top {n_select} pairs")

    logger.info(f"Total final pairs selected: {len(pairs)}")



    # 5) 构建输出并写入 JSONL
    out_dir = os.path.dirname(cfg.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    user_safe = user_unsafe = agent_safe = agent_unsafe = 0
    written = 0

    with open(cfg.output_file, "w", encoding="utf-8") as fout:
        for p in tqdm(pairs, desc="Writing output pairs", unit="pairs"):
            qid = p["qid"]
            item = query_map.get(qid, {})
            prompt_text = item.get("prompt", "")
            chosen_entry = p["chosen"]
            rejected_entry = p["rejected"]

            # candidate objects might include 'text' or other fields; prefer candidate['text'] if present
            chosen_text = chosen_entry.get("candidate", {}).get("text") if isinstance(chosen_entry.get("candidate"), dict) else None
            rejected_text = rejected_entry.get("candidate", {}).get("text") if isinstance(rejected_entry.get("candidate"), dict) else None

            # Fallback: sometimes candidate is already the textual string
            if chosen_text is None:
                chosen_text = chosen_entry.get("candidate", {}).get("response") if isinstance(chosen_entry.get("candidate"), dict) else str(chosen_entry.get("candidate"))
            if rejected_text is None:
                rejected_text = rejected_entry.get("candidate", {}).get("response") if isinstance(rejected_entry.get("candidate"), dict) else str(rejected_entry.get("candidate"))

            rec = {
                "id": qid,
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "chosen_conf": float(p.get("chosen_confidence", 0.0)),
                "rejected_conf": float(p.get("rejected_confidence", 0.0)),
                "confidence_diff": float(p.get("confidence_diff", 0.0))
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

            # update simple stats if we can
            label = item.get("label")
            if label == "harmful":
                if item.get("response") is not None:
                    agent_unsafe += 1
                else:
                    user_safe += 1
            elif label == "unharmful":
                if item.get("response") is not None:
                    agent_safe += 1
                else:
                    user_unsafe += 1

    logger.success(f"Wrote {written} DPO pairs to {cfg.output_file}")

    # 6) 输出统计信息
    logger.stat("safe/usafe-agent/user distribution")
    Table.draw_table(user_safe, user_unsafe, agent_safe, agent_unsafe)

    split_counts = Counter()
    for p in pairs:
        qid = p["qid"]
        item = query_map.get(qid, {})
        split_counts[item.get("split", "unknown")] += 1
    table = [(split, count) for split, count in split_counts.items()]
    table.append(["total", sum(split_counts.values())])
    print(tabulate(table, headers=["Split", "Count"], tablefmt="grid"))

# ================== 主入口 ==================
def main():
    cfg = FilterConfig()

    # 示例：如果需要，可以在这里修改默认 cfg
    # cfg.input_file = "your_input.jsonl"
    # cfg.output_file = "your_output.jsonl"
    # cfg.dpo_mode = "closest"
    # cfg.closest_top_n = 200
    # cfg.threshold_mode = True
    # cfg.chosen_conf_threshold = 0.8
    # cfg.rejected_conf_threshold = 0.2
    # cfg.select_n = 1000

    logger = Logger(verbose=cfg.verbose)
    logger.success("Current config:")
    for k, v in vars(cfg).items():
        logger.info(f"  {k}: {v}")
    filter_candidates(cfg)

if __name__ == "__main__":
    main()
