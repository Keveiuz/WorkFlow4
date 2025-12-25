import os
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal
from tqdm import tqdm
from collections import Counter
from tabulate import tabulate
import numpy as np
import re

# ================== ⚙️ 配置类 ==================
@dataclass
class FilterConfig:
    # 输入 / 输出
    input_file: str = "/workspace/mnt/lxb_work/zez_work/GuardRail/temp/synth_data_benign-inference.jsonl"
    output_file: str = "/workspace/mnt/lxb_work/zez_work/GuardRail/temp/synth_data_benign-filtered.jsonl"

    # 筛选参数（两者都为 None 则只依据 score>=4）
    score_threshold: Literal[1, 2, 3, 4] = 4        # 全局分数阈值，控制回答准确度
    confidence_calc_method: Literal[                # 计算置信度的方法
        "average_trace_confidence", 
        "bottom_percent_group_confidence",
        "group_confidence",
        "lowest_group_confidence",
        "tail_confidence",
        "tail_confidence_by_percent"] = "bottom_percent_group_confidence"
    confidence_threshold: Optional[float] = None    # 全局置信度阈值（ >= 该值）
    select_ratio: Optional[float] = None            # 全局置信度 top N%（0.10 表示前 10%），若为 None 则禁用
    select_n: Optional[int] = 10000                 # 全局置信度 top N（100 表示前100个），若为 None 则禁用
    length_control: Optional[float] = len("<think>\nAnswers:  \nRequest: unharmful  \nCompletion: compliance  \nResponse: unharmful\n</think>\n\n Request: unharmful \nCompletion: compliance \nResponse: unharmful") + 50


    # 新增：选择模式：'top' 或 'bottom'
    selection_mode: Literal["top", "bottom"] = "bottom"

    window_size: int = 1024
    bottom_percent: float = 0.1
    tail_size: int = 1024
    tail_percent: float = 0.1

    # 行为控制
    remove_token_confidence: bool = False            # 是否在最终输出中删除 metrics.token_metrics
    num_candidates: int = 1                         # 使用多少候选进入筛选阶段

# ================== 🔧 工具类 ==================
class Logger():
    def __init__(self):
        self.__GREEN__ = "\033[92m"; self.__YELLOW__ = "\033[93m"; self.__RED__ = "\033[91m"; self.__BLUE__ = "\033[94m"; self.__MAGENTA__ = "\033[95m"; self.__CYAN__ = "\033[96m"; self.__GRAY__ = "\033[90m"; self.__CRITICAL__ = "\033[41;97m"; self.__RESET__ = "\033[0m"
    def success(self, msg): print(f"{self.__GREEN__}[SUCCESS]{self.__RESET__} {msg}")
    def warn(self, msg): print(f"{self.__YELLOW__}[WARNING]{self.__RESET__} {msg}")
    def error(self, msg): print(f"{self.__RED__}[ERROR]{self.__RESET__} {msg}")
    def debug(self, msg): print(f"{self.__BLUE__}[DEBUG]{self.__RESET__} {msg}")
    def stat(self, msg): print(f"{self.__MAGENTA__}[STAT]{self.__RESET__} {msg}")
    def info(self, msg): print(f"{self.__CYAN__}[INFO]{self.__RESET__} {msg}")
    def note(self, msg): print(f"{self.__GRAY__}[NOTE]{self.__RESET__} {msg}")
    def critical(self, msg): print(f"{self.__CRITICAL__}[CRITICAL]{self.__RESET__} {msg}")

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

# ================== 🧮 计算类 ==================
class ConfidenceCalculator:
    """
    计算论文中提到的各种confidence指标
    假设输入是每个token位置的confidence值列表
    """
    
    def __init__(self, token_confidences: List[float]):
        """
        Args:
            token_confidences: 每个token位置的confidence值 [C_0, C_1, ..., C_N]
        """
        self.conf_list = np.array(token_confidences)
        self.n_tokens = len(token_confidences)
    
    # ========== 2. Average Trace Confidence ==========
    def compute_average_trace_confidence(self) -> float:
        return np.mean(self.conf_list)
    
    # ========== 3. Group Confidence ==========
    def compute_group_confidence(self, window_size: int = 2048) -> float:
        conf = self.conf_list
        n = self.n_tokens
        w = window_size
        prefix = np.zeros(n + 1)
        prefix[1:] = np.cumsum(conf)
        left = np.arange(n) - w + 1
        left = np.maximum(left, 0)
        lengths = np.arange(n) - left + 1
        window_sums = prefix[np.arange(1, n + 1)] - prefix[left]
        group_conf = window_sums / lengths
        return np.mean(group_conf)
    
    def compute_group_confidence_positional(self, window_size: int = 2048) -> np.ndarray:
        conf = self.conf_list
        n = self.n_tokens
        w = window_size
        prefix = np.zeros(n + 1)
        prefix[1:] = np.cumsum(conf)
        left = np.arange(n) - w + 1
        left = np.maximum(left, 0)
        lengths = np.arange(n) - left + 1
        window_sums = prefix[np.arange(1, n + 1)] - prefix[left]
        group_conf = window_sums / lengths
        return group_conf
    
    # ========== 4. Bottom X% Group Confidence ==========
    def compute_bottom_percent_group_confidence(self, window_size: int = 2048, bottom_percent: float = 0.1) -> float:
        group_conf = self.compute_group_confidence_positional(window_size=window_size)
        valid_group_conf = group_conf[window_size-1:]
        if len(valid_group_conf) == 0:
            return np.mean(self.conf_list)
        k = max(1, int(len(valid_group_conf) * bottom_percent))
        bottom_k_conf = np.partition(valid_group_conf, k-1)[:k]
        return np.mean(bottom_k_conf)
    
    # ========== 5. Lowest Group Confidence ==========
    def compute_lowest_group_confidence(self, window_size: int = 2048) -> float:
        group_conf = self.compute_group_confidence_positional(window_size)
        valid_group_conf = group_conf[window_size-1:]
        if len(valid_group_conf) == 0:
            return np.min(self.conf_list)
        return np.min(valid_group_conf)
    
    # ========== 6. Tail Confidence ==========
    def compute_tail_confidence(self, tail_size: int = 2048) -> float:
        tail_tokens = self.conf_list[-tail_size:]
        return np.mean(tail_tokens)
    
    def compute_tail_confidence_by_percent(self, tail_percent: float = 0.1) -> float:
        tail_size = max(1, int(self.n_tokens * tail_percent))
        tail_tokens = self.conf_list[-tail_size:]
        return np.mean(tail_tokens)
    
    # ========== 综合报告 ==========
    def compute_all_metrics(self, window_size: int = 2048, bottom_percent: float = 10.0, tail_size: int = 2048) -> dict:
        metrics = {
            'average_trace_conf': self.compute_average_trace_confidence(),
            'bottom_10_group_conf': self.compute_bottom_percent_group_confidence(window_size, bottom_percent),
            'lowest_group_conf': self.compute_lowest_group_confidence(window_size),
            'tail_conf_fixed': self.compute_tail_confidence(tail_size),
            'tail_conf_10percent': self.compute_tail_confidence_by_percent(0.1),
            'n_tokens': self.n_tokens,
        }
        metrics['group_conf_array'] = self.compute_group_confidence_positional(window_size)
        return metrics

# ================== 模板 ==================
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

# ================== 判断回答是否正确 ==================
def extract_labels_from_text(text: str):
    import re
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
        value = value.lower().strip()
        if value in ["harmful"]:
            return "harmful"
        if value in ["unharmful", "harmless", "not harmful"]:
            return "unharmful"
        if value in ["refusal", "compliance"]:
            return value
        if value in ["none", "null", "empty"]:
            return None
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

    return outside_text, answer_dict, think_content

def compute_score(item, answer_dict):
    prompt_harm_label = item["prompt_harm_label"]
    response_refusal_label = item["response_refusal_label"]
    response_harm_label = item["response_harm_label"]
    score = 0 
    if answer_dict == None: return score
    score = 1
    if answer_dict["prompt_harm"] != prompt_harm_label: return score
    score = 2
    if answer_dict["response_harm"] != response_harm_label: return score
    score = 3
    if answer_dict["response_refusal"] != response_refusal_label: return score
    score = 4
    return score

# ================== 主逻辑 ==================
def filter_candidates():
    logger = Logger()
    logger.info(f"Loading data from {FilterConfig.input_file}")
    all_data = []

    # 1) 读取文件
    try:
        total_lines = sum(1 for _ in open(FilterConfig.input_file, "r", encoding="utf-8"))
        with open(FilterConfig.input_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc="Reading input file", unit="lines"):
                line = line.strip()
                if not line:
                    continue
                try:
                    all_data.append(json.loads(line))
                except Exception as e:
                    logger.warn(f"[WARN] failed to parse line: {e}")
    except FileNotFoundError:
        logger.info(f"Input file not found: {FilterConfig.input_file}")
        return

    logger.info(f"Total queries loaded: {len(all_data)}")
    query_map = { item["id"]: item for item in all_data }

    # 2) 对 candidate 做 score 检查，并计算置信度
    valid_candidates = []
    valid_query_ids = set()

    for item in tqdm(all_data, desc="Validating candidates", unit="queries"):
        qid = item["id"]
        for cand in item["candidates"][:FilterConfig.num_candidates]:
            resp_text = cand["text"]

            # 先筛选长度合格的 candidate（杜绝没有推理只有答案的情况）
            if FilterConfig.length_control is not None and len(resp_text) < FilterConfig.length_control:
                continue

            _, ans, _ = extract_labels_from_text(resp_text)
            score = compute_score(item, ans)
            if score >= FilterConfig.score_threshold:
                token_conf = cand["token_confidence"]
                conf_calc = ConfidenceCalculator(token_confidences=token_conf)
                
                if FilterConfig.confidence_calc_method == "average_trace_confidence":
                    conf = conf_calc.compute_average_trace_confidence()
                elif FilterConfig.confidence_calc_method == "bottom_percent_group_confidence":
                    conf = conf_calc.compute_bottom_percent_group_confidence(window_size=FilterConfig.window_size, bottom_percent=FilterConfig.bottom_percent)
                elif FilterConfig.confidence_calc_method == "group_confidence":
                    conf = conf_calc.compute_group_confidence(window_size=FilterConfig.window_size)
                elif FilterConfig.confidence_calc_method == "lowest_group_confidence":
                    conf = conf_calc.compute_lowest_group_confidence(window_size=FilterConfig.window_size)
                elif FilterConfig.confidence_calc_method == "tail_confidence":
                    conf = conf_calc.compute_tail_confidence(tail_size=FilterConfig.tail_size)
                elif FilterConfig.confidence_calc_method == "tail_confidence_by_percent":
                    conf = conf_calc.compute_tail_confidence_by_percent(tail_percent=FilterConfig.tail_percent)
                else:
                    raise TypeError("unsupported confidence calculate method! supported method include: [\"average_trace_confidence\", \"bottom_percent_group_confidence\", \"group_confidence\", \"lowest_group_confidence\", \"tail_confidence\", \"tail_confidence_by_percent\"]")

                conf_val = float(conf) if conf is not None else float("-inf")
                valid_candidates.append({
                    "query_id": qid,
                    "candidate": cand,
                    "confidence": conf_val,
                    "token_confidence": token_conf,
                })
                valid_query_ids.add(qid)

    logger.info(f"Total {len(valid_candidates)} candidates are qulified.")
    logger.info(f"Total {len(valid_query_ids)} queries with at least one qulified candidate")

    if not valid_candidates:
        logger.info("No fully correct candidates. Exiting.")
        return

    # 3) 按 confidence_threshold 过滤（按 query 取一个）
    if FilterConfig.confidence_threshold is not None:
        selected_by_query = {}
        for entry in valid_candidates:
            qid = entry["query_id"]
            conf = entry["confidence"]
            if conf < FilterConfig.confidence_threshold:
                continue
            if qid in selected_by_query:
                continue
            selected_by_query[qid] = entry
        logger.info(f"Applied confidence_threshold={FilterConfig.confidence_threshold}: {len(selected_by_query)} queries selected")
        # 更新 valid_candidates 为最终筛选结果（列表形式）
        valid_candidates = list(selected_by_query.values())

    # 4) 按全局 top N% / top N / bottom N% / bottom N 过滤
    # 说明：
    # - selection_mode 决定排序方向：'top' -> 降序（高置信度优先）；'bottom' -> 升序（低置信度优先）
    # - select_ratio 优先，如果为 None 则使用 select_n；若两者都为 None 则不进行此全局筛选（继续按每 query 选最佳）
    if FilterConfig.select_ratio is not None or FilterConfig.select_n is not None:
        # 先去重计算 qualified queries 总数（基于当前 valid_candidates）
        all_query_ids_in_valid = list(dict.fromkeys([c["query_id"] for c in valid_candidates]))
        total_queries = len(set(all_query_ids_in_valid))
        # decide sort order
        reverse = True if FilterConfig.selection_mode == "top" else False
        logger.info(f"Selection mode: {FilterConfig.selection_mode} (reverse={reverse})")

        # 按 confidence 排序（整个候选集合），然后按 query 去重取第一个（保证每 query 最多一个）
        sorted_candidates = sorted(valid_candidates, key=lambda x: x["confidence"], reverse=reverse)

        if FilterConfig.select_ratio is not None:
            # 以 query 总数来计算要选的 queries 数量（更合理）
            n_queries_to_select = max(1, int(total_queries * FilterConfig.select_ratio))
            logger.info(f"Applied select_ratio={FilterConfig.select_ratio*100:.2f}% -> plan to collect {n_queries_to_select} queries (from {total_queries} qualified queries).")
        else:
            n_queries_to_select = max(1, min(FilterConfig.select_n, total_queries))
            logger.info(f"Applied select_n={FilterConfig.select_n} -> plan to collect {n_queries_to_select} queries (from {total_queries} qualified queries).")

        selected_by_query = {}
        for entry in sorted_candidates:
            qid = entry["query_id"]
            if qid not in selected_by_query:
                selected_by_query[qid] = entry
                if len(selected_by_query) >= n_queries_to_select:
                    break

        logger.info(f"Selected {len(selected_by_query)} queries out of {total_queries} qualified queries using mode={FilterConfig.selection_mode}")
        # 更新 valid_candidates（列表形式）
        valid_candidates = list(selected_by_query.values())
    else:
        # 如果不使用 select_ratio/select_n，则按每个 query 选择 confidence 最高（若 selection_mode == 'bottom'，则选择最低）
        selected_by_query = {}
        for entry in valid_candidates:
            qid = entry["query_id"]
            conf = entry["confidence"]
            prev = selected_by_query.get(qid)
            if prev is None:
                selected_by_query[qid] = entry
            else:
                # 如果 mode 是 top，保留更高的；如果 mode 是 bottom，保留更低的
                if FilterConfig.selection_mode == "top":
                    if conf > prev["confidence"]:
                        selected_by_query[qid] = entry
                else:
                    if conf < prev["confidence"]:
                        selected_by_query[qid] = entry

    logger.info(f"Queries that have at least one selected candidate: {len(selected_by_query)}")

    # 5) 构建最终输出记录
    filtered_items = []
    user_safe = user_unsafe = agent_safe = agent_unsafe = 0

    for qid, sel in tqdm(selected_by_query.items(), desc="Building final records", unit="queries"):
        item = query_map.get(qid)
        if item is None:
            continue
        cand = sel["candidate"]
        conf = sel["confidence"]
        token_confidence = sel["token_confidence"]

        _, _, reasoning_trace = extract_labels_from_text(cand["text"])

        user_query = REASONING_TEMPLATE.format(prompt=item['prompt'], response=item['response'])
        assistant_response = f"<think>{reasoning_trace}</think> Request: {item['prompt_harm_label']} \nCompletion: {item['response_refusal_label']} \nResponse: {item['response_harm_label']}"

        conversations = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_response},
        ]
        if item['label'] == "harmful":
            if item['response'] is not None:
                agent_unsafe += 1
            else:
                user_safe += 1
        elif item['label'] == "unharmful":
            if item['response'] is not None:
                agent_safe += 1
            else:
                user_unsafe += 1
        
        if FilterConfig.remove_token_confidence:
            rec = {
                "id": item['id'],
                "split": item['split'],
                "conversations": conversations,
                "user": item['prompt'],
                "assistant": item['response'],
                "label": item['label'],
                "prompt_harm_label": item['prompt_harm_label'],
                "response_refusal_label": item['response_refusal_label'],
                "response_harm_label": item['response_harm_label'],
                "selected_confidence": conf
            }
        else:
            rec = {
                "id": item['id'],
                "split": item['split'],
                "conversations": conversations,
                "user": item['prompt'],
                "assistant": item['response'],
                "label": item['label'],
                "prompt_harm_label": item['prompt_harm_label'],
                "response_refusal_label": item['response_refusal_label'],
                "response_harm_label": item['response_harm_label'],
                "token_confidence": token_confidence,
                "selected_confidence": conf
            }
        filtered_items.append(rec)
    # 按 id 从小到大排序
    filtered_items.sort(key=lambda x: x["id"])

    logger.info(f"Total queries selected (final): {len(filtered_items)}")

    # 6) 保存到输出 JSONL
    out_dir = os.path.dirname(FilterConfig.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(FilterConfig.output_file, "w", encoding="utf-8") as fout:
        for rec in tqdm(filtered_items, desc="Writing output file", unit="records"):
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Saved filtered SFT data to {FilterConfig.output_file}")

    # 7) 输出统计信息
    logger.stat("safe/usafe-agent/user distribution")
    Table.draw_table(user_safe, user_unsafe, agent_safe, agent_unsafe)

    split_counts = Counter(r["split"] for r in filtered_items)
    table = [(split, count) for split, count in split_counts.items()]
    total_selected = sum(split_counts.values())
    table.append(["total", total_selected])
    print(tabulate(table, headers=["Split", "Count"], tablefmt="grid"))


# ================== 主入口 ==================
def main():
    cfg = FilterConfig()  # 修改这里的配置以改变行为
    logger = Logger()
    logger.success("Current config:")
    for k, v in vars(cfg).items():
        logger.success(f"  {k}: {v}")
    filter_candidates()

if __name__ == "__main__":
    main()
