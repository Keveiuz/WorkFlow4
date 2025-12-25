import matplotlib.pyplot as plt

# --- 数据定义 ---
x_labels = ['high_high', 'high_low', 'low_high', 'low_low']
# 您的四个数据点
data_scores = [76.09, 76.72, 76.42, 76.63]
# 基准线值
baseline_value = 76.87 

# --- 绘图设置 ---
plt.figure(figsize=(8, 6))

# 1. 绘制数据折线图 (使用蓝色实线和圆点标记)
plt.plot(x_labels, data_scores, 
         marker='o', 
         linestyle='-', 
         color='tab:blue', 
         label='Model Scores')

# 2. 绘制基准线 (水平于x轴的灰色虚线，并添加标记以增强可见性)
# 使用 plt.axhline 绘制水平线，它不需要 x_labels
plt.axhline(y=baseline_value, 
            color='gray', 
            linestyle='--', 
            linewidth=1.5, 
            label=f'Baseline ({baseline_value})')

# 3. 设置图表属性
plt.title('Performance Comparison Across Different Settings')
plt.xlabel('Configuration')
plt.ylabel('Score (%)')

# 确保 Y 轴范围聚焦在数据附近
min_score = min(min(data_scores), baseline_value)
max_score = max(max(data_scores), baseline_value)
plt.ylim(min_score - 0.05, max_score + 0.02) # 稍微扩展Y轴范围以美观

plt.grid(True, linestyle=':', alpha=0.6) # 添加网格线
plt.legend() # 显示图例

# 4. 保存图表
file_name = 'performance_comparison.png' # 您希望的文件名
plt.savefig(file_name, dpi=300, bbox_inches='tight') 
print(f"图表已保存为 {file_name}")
# plt.close() # 可选：保存后关闭图形，释放内存
