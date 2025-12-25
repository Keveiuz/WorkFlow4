import matplotlib.pyplot as plt
import numpy as np

# 1. 定义数据和分组
x_labels = ['10K', '20K', '40K', '80K']

# 定义新的基准线数据
baseline_scores = [76.07, 76.62, 76.41, 76.49]

all_scores = [
    76.45, 75.92, 76.34, 76.72,  # 0: average_trace_confidence-top
    73.13, 76.42, 76.62, 76.69,  # 1: average_trace_confidence-bottom
    76.36, 76.87, 76.40, 76.50,  # 2: bottom_percent_group_confidence-top
    72.44, 75.73, 76.48, 75.98,  # 3: bottom_percent_group_confidence-bottom
    75.95, 76.51, 75.84, 76.88,  # 4: tail_confidence_by_percent-top
    71.21, 75.50, 75.92, 76.30   # 5: tail_confidence_by_percent-bottom
]

group_names = [
    'average_trace_confidence-top',
    'average_trace_confidence-bottom',
    'bottom_percent_group_confidence-top',
    'bottom_percent_group_confidence-bottom',
    'tail_confidence_by_percent-top',
    'tail_confidence_by_percent-bottom'
]

data = [all_scores[i:i + 4] for i in range(0, len(all_scores), 4)]

# 定义颜色映射 (C0, C1, C2, ...)
colors = [f'C{i}' for i in range(6)]

# 定义 Top/Bottom 对比子图的配置
comparison_configs = [
    (0, 1, 'Average Trace Confidence'),          # 子图 2 (右上)
    (2, 3, 'Bottom Percent Group Confidence'),   # 子图 3 (左下)
    (4, 5, 'Tail Confidence by Percent')         # 子图 4 (右下)
]

# 2. 绘制 2x2 子图
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

# 设置全局Y轴范围
min_score = min(min(all_scores), min(baseline_scores)) # 考虑 baseline 调整范围
max_score = max(max(all_scores), max(baseline_scores))
y_limit = (min_score - 0.5, max_score + 0.5)


# --- 子图 1 (左上): 所有折线 (ax1) ---
ax1 = axes[0, 0]
for i in range(len(data)):
    ax1.plot(x_labels, data[i],
             marker='o', linestyle='-', color=colors[i], 
             label=group_names[i])

# **添加基准线**
ax1.plot(x_labels, baseline_scores,
         marker='', linestyle='--', color='gray', 
         label='Baseline')

ax1.set_title('Overall Performance (All 6 Metrics)', fontsize=14)
ax1.set_ylabel('Score')
ax1.set_ylim(y_limit)
ax1.grid(True, linestyle=':', alpha=0.7)
# 图例放在右下角
ax1.legend(loc='lower right', fontsize=9, frameon=True)


# --- 子图 2, 3, 4: Top/Bottom 对比 ---
for i, (top_idx, bottom_idx, title) in enumerate(comparison_configs):
    row = (i + 1) // 2
    col = (i + 1) % 2
    ax = axes[row, col]
    
    # 绘制 Top 组
    ax.plot(x_labels, data[top_idx], 
            marker='o', linestyle='-', color=colors[top_idx], 
            label=group_names[top_idx])
    
    # 绘制 Bottom 组
    ax.plot(x_labels, data[bottom_idx], 
            marker='s', linestyle='-', color=colors[bottom_idx], 
            label=group_names[bottom_idx])
            
    # **添加基准线**
    ax.plot(x_labels, baseline_scores,
             marker='^', linestyle='--', color='gray', 
             label='Baseline')
    
    # 设置标题、Y轴范围、网格
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Score')
    ax.set_ylim(y_limit)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # 图例放在右下角
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    
    # 仅在底部的子图上显示X轴标签
    if row == 1:
        ax.set_xlabel('Data Size (K)')

# 调整子图间距和布局
plt.tight_layout()

# 3. 保存图形为 PNG 文件
output_filename = 'scores_4_subplots_with_baseline.png'
plt.savefig(output_filename, dpi=300)

plt.close()

print(f"✅ 图表已成功保存为文件: {output_filename}")


# import matplotlib.pyplot as plt
# import numpy as np

# # 1. 定义数据
# x_labels = ['10K', '20K', '40K', '80K']

# all_scores = [
#     76.45, 75.92, 76.34, 76.72,
#     73.13, 76.42, 76.62, 76.69,
#     76.36, 76.87, 76.40, 76.50,
#     72.44, 75.73, 76.48, 75.98,
#     75.95, 76.51, 75.84, 76.88,
#     71.21, 75.50, 75.92, 76.30
# ]

# group_names = [
#     'average_trace_confidence-top',
#     'average_trace_confidence-bottom',
#     'bottom_percent_group_confidence-top',
#     'bottom_percent_group_confidence-bottom',
#     'tail_confidence_by_percent-top',
#     'tail_confidence_by_percent-bottom'
# ]

# data = [all_scores[i:i + 4] for i in range(0, len(all_scores), 4)]

# # 2. 绘制折线图
# plt.figure(figsize=(10, 6))

# for i in range(len(data)):
#     plt.plot(x_labels, data[i],
#              marker='o',
#              linestyle='-',
#              label=group_names[i])

# # 3. 添加图表元素
# plt.title('Performance Scores Across Different Data Sizes (10K to 80K)')
# plt.xlabel('Data Size (K)')
# plt.ylabel('Score')

# plt.grid(True, linestyle='--', alpha=0.6)

# # *** 关键修改：将图例放置在图表内部，使用 loc='best' 让 Matplotlib 自动选择不遮挡线条的最佳位置。***
# plt.legend(title='Metrics', loc='best', frameon=True) # frameon=True 保持图例的边框，使其更清晰

# # 调整Y轴范围
# min_score = min(all_scores)
# max_score = max(all_scores)
# plt.ylim(min_score - 1, max_score + 0.5)

# # 使用 tight_layout 自动调整布局，因为图例现在在内部，所以不需要 rect 参数
# plt.tight_layout()

# # 4. 保存图形为 PNG 文件
# output_filename = 'scores_line_chart_legend_inside.png'
# plt.savefig(output_filename, dpi=300)

# plt.close()

# print(f"✅ 图表已成功保存为文件: {output_filename}")




# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # ==================== 1. 数据定义 ====================
# # 15个标签 (用于定义分组和横轴)
# labels = [
#     "dpo-average_trace_confidence-closest", "dpo-average_trace_confidence-high_high", 
#     "dpo-average_trace_confidence-high_low", "dpo-average_trace_confidence-low_high", 
#     "dpo-average_trace_confidence-low_low", "dpo-bottom_percent_group_confidence-closest", 
#     "dpo-bottom_percent_group_confidence-high_high", "dpo-bottom_percent_group_confidence-high_low", 
#     "dpo-bottom_percent_group_confidence-low_high", "dpo-bottom_percent_group_confidence-low_low", 
#     "dpo-tail_confidence_by_percent-closest", "dpo-tail_confidence_by_percent-high_high", 
#     "dpo-tail_confidence_by_percent-high_low", "dpo-tail_confidence_by_percent-low_high", 
#     "dpo-tail_confidence_by_percent-low_low"
# ]
# # 15个平均分数 (纵轴数据)
# scores = [
#     79.14, 79.21, 78.99, 79.28, 79.21,
#     79.14, 79.13, 79.09, 79.17, 79.15,
#     79.16, 79.23, 79.25, 79.14, 79.09
# ]

# # 转换为 DataFrame
# df = pd.DataFrame({'Label': labels, 'Score': scores})

# # ==================== 2. 数据处理与分组 ====================
# # 拆分标签为 [方法, 置信度指标, 边界策略]
# df[['Method', 'Metric', 'Strategy']] = df['Label'].str.split('-', expand=True)

# # 创建数据透视表：索引为 Strategy (横轴), 列为 Metric (三条线), 值为 Score (纵轴)
# pivot_df = df.pivot(index='Strategy', columns='Metric', values='Score')

# # 定义横轴的显示顺序
# strategy_order = ['closest', 'high_high', 'high_low', 'low_high', 'low_low']
# pivot_df = pivot_df.reindex(strategy_order)

# # ==================== 3. 绘图设置 ====================
# plt.figure(figsize=(10, 6))

# # 绘制折线图，每条线代表一个置信度指标
# for column in pivot_df.columns:
#     # 格式化图例名称 (例如: average_trace_confidence -> Average Trace Confidence)
#     legend_label = column.replace('_', ' ').title()
#     plt.plot(pivot_df.index, pivot_df[column], marker='o', label=legend_label, linewidth=2)

# # 设置图表标题和标签
# plt.title('Average Score by Boundary Strategy and Confidence Metric')
# plt.xlabel('Boundary Strategy')
# plt.ylabel('Average Score')

# # 优化 Y 轴显示范围 (突出数值波动)
# y_min = pivot_df.min().min() - 0.05
# y_max = pivot_df.max().max() + 0.05
# plt.ylim(y_min, y_max)
# plt.yticks(np.arange(np.floor(y_min * 100) / 100, np.ceil(y_max * 100) / 100, 0.05))

# # 旋转 X 轴标签，提高可读性
# plt.xticks(rotation=15, ha='right')

# # 添加网格
# plt.grid(True, linestyle='--', alpha=0.6)

# # 将图例放在图表内部，使用 loc 参数进行定位
# # 例如：
# # loc='upper left' (左上角)
# # loc='upper right' (右上角)
# # loc='lower left' (左下角)
# # loc='lower right' (右下角)
# # loc='best' (Matplotlib 会自动选择最佳位置，通常是默认值)
# plt.legend(title='Confidence Metric', loc='upper left') 

# # 由于图例在内部，不再需要调整 rect 参数
# plt.tight_layout()

# # ==================== 4. 输出图表 ====================
# plt.savefig('average_score_line_chart.png', dpi=300)
# print("average_score_line_chart.png")
