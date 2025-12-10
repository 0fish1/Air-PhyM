import pickle
import numpy as np

# --- 修改为您数据文件的路径 ---
PKL_FILE_PATH = "/home/yy/pollution_mul/code/data_deal/data/bj/samples_48h.pkl"

# 加载数据
with open(PKL_FILE_PATH, "rb") as f:
    samples = pickle.load(f)

# 提取所有样本的真实标签（PM2.5浓度值）
labels = np.array([sample['target'] for sample in samples]).flatten()

print(f"总共加载了 {len(labels)} 个样本。")
print(f"PM2.5 浓度范围: 从 {np.min(labels):.2f} 到 {np.max(labels):.2f}")

# 找到低、中、高污染值的样本索引
low_pollution_idx = np.argmin(labels)
high_pollution_idx = np.argmax(labels)

# 找到中位数的索引
median_value = np.median(labels)
# 找到离中位数最近的样本的索引
median_idx = np.abs(labels - median_value).argmin()


print("\n--- 推荐的锚点索引 ---")
print(f"低污染锚点 (值: {labels[low_pollution_idx]:.2f}): --anchor_idx {low_pollution_idx}")
print(f"中污染锚点 (值: {labels[median_idx]:.2f}): --anchor_idx {median_idx}")
print(f"高污染锚点 (值: {labels[high_pollution_idx]:.2f}): --anchor_idx {high_pollution_idx}")

# 注意：为了让可视化脚本能够选中这些索引，您需要确保 --batch_size 足够大
# 例如，如果高污染索引是 150，您的 batch_size 至少要大于 150。
# 或者，您可以修改可视化脚本的数据加载方式，直接加载指定索引的样本。
# 但最简单的方法是设置一个足够大的batch_size。