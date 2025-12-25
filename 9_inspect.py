import json

# --- 请修改为您实际的文件路径 ---
file_path = "/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/guardreasoner-metric.jsonl" 
target_id = 101266
# ----------------------------------

print(f"正在文件中查找 ID: {target_id} 的数据...")

try:
    found = False
    with open(file_path, 'r', encoding='utf-8') as f:
        # 逐行读取文件，适合处理大型 JSONL 文件
        for line_number, line in enumerate(f, 1):
            try:
                # 解析 JSON 对象
                data = json.loads(line)
                
                # 检查 ID 字段。注意：这里假设 ID 是键 'id'
                if data.get('id') == target_id:
                    print("-" * 30)
                    print(f"✅ 在第 {line_number} 行找到了目标数据:")
                    # 使用 json.dumps 格式化打印结果，使其更易读
                    print(json.dumps(data, indent=4, ensure_ascii=False))
                    print("-" * 30)
                    found = True
                    break # 找到后即停止查找
                    
            except json.JSONDecodeError:
                print(f"⚠️ 警告: 第 {line_number} 行不是有效的 JSON 格式，已跳过。")
    
    if not found:
        print(f"❌ 查找完毕，文件中未找到 ID 为 {target_id} 的数据。")

except FileNotFoundError:
    print(f"❌ 错误: 找不到文件路径 '{file_path}'，请检查路径是否正确。")
except Exception as e:
    print(f"发生未知错误: {e}")
