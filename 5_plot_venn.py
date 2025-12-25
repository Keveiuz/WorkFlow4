import json
from matplotlib import pyplot as plt
from matplotlib_venn import venn2 # ⚠️ 关键修改：从 venn3 变为 venn2
from typing import Set
import pandas as pd
import os

# --- 1. 数据读取和 ID 提取函数 (保持不变) ---
def load_ids_from_jsonl(filepath: str) -> Set[str]:
    """
    从 JSONL 文件中读取数据，并提取所有 'id' 字段的值。
    """
    if not os.path.exists(filepath):
        print(f"⚠️ 警告：文件未找到: {filepath}。返回空集合。")
        return set()
    
    ids = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'id' in data:
                            # 确保 ID 是字符串类型
                            ids.add(str(data['id']))
                    except json.JSONDecodeError:
                        print(f"❌ 错误：跳过无法解析的行: {line.strip()[:50]}...")
                    except Exception as e:
                        print(f"❌ 错误：处理行时发生异常: {e}")
    except Exception as e:
        print(f"❌ 错误：读取文件 {filepath} 时发生异常: {e}")

    print(f"✅ 成功从 {os.path.basename(filepath)} 中提取 {len(ids)} 个唯一 ID。")
    return ids

# --- 2. 主绘图函数 (已修改为两个集合) ---
def plot_venn_diagram_2d(file1_path: str, file2_path: str, title: str = "Different Data Filtered by 2 Confidence Metrics") -> None:
    
    # 定义集合名称
    label1 = "top-confidence"
    label2 = "bottom-confidence"
    
    # 提取 ID 集合
    set1 = load_ids_from_jsonl(file1_path)
    set2 = load_ids_from_jsonl(file2_path)
    
    # 集合为空检查
    if not (set1 or set2):
        print("所有文件都没有有效 ID，无法绘制韦恩图。")
        return

    # 绘制韦恩图
    plt.figure(figsize=(8, 8))
    
    # venn2 需要 3 个数字来表示 3 个区域的大小：
    # (Ab, aB, AB)
    # 区域含义：
    # Ab: 仅在 set1 中
    # aB: 仅在 set2 中
    # AB: 在 set1 和 set2 中
    
    venn = venn2(
        subsets=(set1, set2), # ⚠️ 关键修改：只传入两个集合
        set_labels=(label1, label2)
    )

    # 设置标题
    plt.title(title, fontsize=16)
    
    # === 保存图片的代码 (保持不变) ===
    output_filename = "venn_diagram_2d_output.png"
    try:
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        print(f"🖼️ 韦恩图已成功保存到本地文件: {output_filename}")
    except Exception as e:
        print(f"❌ 错误：保存图片失败: {e}。图形将关闭。")
    finally:
        plt.close()
    # ==========================
     
    # --- 额外的文本输出：详细的交集/差集数量 (已修改) ---
    print("\n--- 详细交集/差集分析 ---")
    
    # 计算交集和差集
    only_set1 = set1 - set2
    only_set2 = set2 - set1
    intersection = set1 & set2
    union_set = set1 | set2
    
    results = [
        (f"仅在 A ({label1}) 中", len(only_set1)),
        (f"仅在 B ({label2}) 中", len(only_set2)),
        ("A 和 B 共有 (交集)", len(intersection)),
        ("总的唯一 ID 数量 (并集)", len(union_set)),
    ]
    
    df = pd.DataFrame(results, columns=["区域", "ID 数量"])
    print(df.to_markdown(index=False))

# --- 3. 示例用法 (已修改为只使用两个文件) ---
if __name__ == "__main__":
    # ⚠️ 请修改为您的 JSONL 文件路径 ⚠️
    FILE_A = "/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/guardreasoner-filtered-top.jsonl"
    FILE_B = "/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/guardreasoner-filtered-bottom.jsonl"
    
    plot_venn_diagram_2d(FILE_A, FILE_B)
