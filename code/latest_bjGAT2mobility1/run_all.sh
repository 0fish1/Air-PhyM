#!/bin/bash

# 设置锚点索引的范围
START_IDX=1
END_IDX=200

echo "开始批量生成可视化图，从索引 ${START_IDX} 到 ${END_IDX}..."

# 使用 for 循环遍历范围
for (( i=${START_IDX}; i<=${END_IDX}; i++ ))
do
  echo "--- 正在处理 anchor_idx = ${i} ---"
  # 执行你的 Python 脚本，并将索引 i 作为参数传入
  python Visualization6.py --anchor_idx ${i}
done

echo "所有任务已完成！"