import json
import pandas as pd
from tqdm import tqdm

def format_dialogue(prompt, answer):
    """
    将 prompt + answer 拼成对话格式：
    [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer}
    ]
    """
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer}
    ]

def jsonl_to_parquet(jsonl_path, parquet_path):

    ids = []
    chosen_dialogs = []
    rejected_dialogs = []

    print(f"Loading {jsonl_path} ...")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Converting"):
            obj = json.loads(line)

            prompt = obj["prompt"]
            chosen_raw = obj["chosen"]
            rejected_raw = obj["rejected"]

            ids.append(obj["id"])
            chosen_dialogs.append(format_dialogue(prompt, chosen_raw))
            rejected_dialogs.append(format_dialogue(prompt, rejected_raw))

    df = pd.DataFrame({
        "id": ids,
        "chosen": chosen_dialogs,
        "rejected": rejected_dialogs
    })

    df.to_parquet(parquet_path, index=False)
    print(f"Saved parquet to {parquet_path}")

if __name__ == "__main__":
    jsonl_to_parquet(
        jsonl_path="/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/guardreasoner-preference.jsonl",
        parquet_path="/workspace/mnt/lxb_work/zez_work/Boundary-Sample/train/dpo-bottom_percent_group_confidence-low_high.parquet")
