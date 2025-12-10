import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

# 导入项目模块
from model import AirQualityModel
from dataset import AirQualityDataset
from configs import experiment_configs

# get_all_embeddings 函数保持不变

def plot_full_anchor_view(ax, tsne_img, tsne_num, labels, anchor_idx, 
                          cross_modal_pos, intra_modal_pos,
                          negative_indices, distances, title):
    """
    绘制包含同模态和跨模态邻居的完整锚点视图。
    """
    ax.scatter(tsne_img[:, 0], tsne_img[:, 1], c='gray', alpha=0.1, marker='o', label='_nolegend_')
    ax.scatter(tsne_num[:, 0], tsne_num[:, 1], c='gray', alpha=0.1, marker='x', label='_nolegend_')

    anchor_pos = tsne_img[anchor_idx]
    anchor_label = labels[anchor_idx]
    
    ax.scatter(anchor_pos[0], anchor_pos[1], c=[anchor_label], cmap='viridis',
               marker='*', s=400, edgecolor='black', zorder=5, label=f'锚点 (图像, PM2.5={anchor_label:.1f})')

    # --- 绘制跨模态正样本 (时空特征) ---
    points = tsne_num[cross_modal_pos]
    ax.scatter(points[:, 0], points[:, 1], c=labels[cross_modal_pos], cmap='viridis',
               marker='o', s=150, edgecolor='green', linewidth=2, zorder=4, label='正样本 (跨模态)')

    # --- 绘制同模态正样本 (图像特征) ---
    points = tsne_img[intra_modal_pos]
    ax.scatter(points[:, 0], points[:, 1], c=labels[intra_modal_pos], cmap='viridis',
               marker='s', s=150, edgecolor='blue', linewidth=2, zorder=4, label='正样本 (同模态)')

    # --- 绘制负样本 (为了简化，我们只显示时空负样本) ---
    points = tsne_num[negative_indices]
    ax.scatter(points[:, 0], points[:, 1], c=labels[negative_indices], cmap='viridis',
               marker='X', s=150, edgecolor='red', linewidth=2, zorder=4, label='负样本 (跨模态)')

    # --- 绘制连线并标注距离 ---
    # 跨模态正样本连线 (绿色实线)
    for idx in cross_modal_pos:
        p_pos = tsne_num[idx]
        dist = distances['cross_modal'][idx]
        ax.plot([anchor_pos[0], p_pos[0]], [anchor_pos[1], p_pos[1]], 'g-', alpha=0.7)
        mid_point = (anchor_pos + p_pos) / 2
        ax.text(mid_point[0], mid_point[1], f'{dist:.2f}', color='darkgreen', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

    # 同模态正样本连线 (蓝色点线)
    for idx in intra_modal_pos:
        p_pos = tsne_img[idx]
        dist = distances['intra_modal'][idx]
        ax.plot([anchor_pos[0], p_pos[0]], [anchor_pos[1], p_pos[1]], 'b:', alpha=0.7)
        mid_point = (anchor_pos + p_pos) / 2
        ax.text(mid_point[0], mid_point[1], f'{dist:.2f}', color='darkblue', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

    # 负样本连线 (红色虚线)
    for idx in negative_indices:
        n_pos = tsne_num[idx]
        dist = distances['cross_modal'][idx] # 距离是相对于时空特征计算的
        ax.plot([anchor_pos[0], n_pos[0]], [anchor_pos[1], n_pos[1]], 'r--', alpha=0.6)
        mid_point = (anchor_pos + n_pos) / 2
        ax.text(mid_point[0], mid_point[1], f'{dist:.2f}', color='darkred', fontsize=9, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == '__main__':
    # ... (参数解析部分保持不变) ...
    # 为了简洁，此处省略，请使用上一个脚本的参数解析部分
    
    # ... (设备选择、数据和模型加载部分保持不变) ...
    # 为了简洁，此处省略

    # --- 核心分析逻辑修改 ---
    results = {}
    for model_name, (img_embeds, num_embeds) in {
        "PM": (img_embeds_pm, num_embeds_pm),
        "PM-SCL": (img_embeds_scl, num_embeds_scl)
    }.items():
        print(f"正在分析模型: {model_name}")
        
        img_embeds_norm = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
        num_embeds_norm = num_embeds / np.linalg.norm(num_embeds, axis=1, keepdims=True)

        anchor_embed = img_embeds_norm[args.anchor_idx:args.anchor_idx+1]
        anchor_label = labels[args.anchor_idx]

        # --- 计算两种距离 ---
        cross_modal_distances = cdist(anchor_embed, num_embeds_norm, 'euclidean').flatten()
        intra_modal_distances = cdist(anchor_embed, img_embeds_norm, 'euclidean').flatten()
        
        all_indices = np.arange(len(labels))
        candidate_indices = all_indices[all_indices != args.anchor_idx]
        
        label_diffs = np.abs(labels[candidate_indices] - anchor_label)
        
        positive_mask = label_diffs < args.positive_threshold
        negative_mask = ~positive_mask

        possible_pos_indices = candidate_indices[positive_mask]
        possible_neg_indices = candidate_indices[negative_mask]
        
        # --- 分别寻找两种正样本 ---
        # 1. 跨模态正样本 (时空)
        pos_cross_distances = cross_modal_distances[possible_pos_indices]
        sorted_cross_indices = possible_pos_indices[np.argsort(pos_cross_distances)]
        top_k_cross_positives = sorted_cross_indices[:args.num_positives]

        # 2. 同模态正样本 (图像)
        pos_intra_distances = intra_modal_distances[possible_pos_indices]
        sorted_intra_indices = possible_pos_indices[np.argsort(pos_intra_distances)]
        top_k_intra_positives = sorted_intra_indices[:args.num_positives]

        # --- 寻找负样本 (以跨模态距离为标准) ---
        neg_cross_distances = cross_modal_distances[possible_neg_indices]
        sorted_neg_indices = possible_neg_indices[np.argsort(neg_cross_distances)]
        top_m_negatives = sorted_neg_indices[-args.num_negatives:]

        # t-SNE... (保持不变)

        results[model_name] = {
            "tsne_img": tsne_img, "tsne_num": tsne_num,
            "cross_modal_pos": top_k_cross_positives,
            "intra_modal_pos": top_k_intra_positives,
            "negative_indices": top_m_negatives,
            "distances": {"cross_modal": cross_modal_distances, "intra_modal": intra_modal_distances}
        }

    # --- 可视化调用修改 ---
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    
    plot_full_anchor_view(axes[0], results['PM']['tsne_img'], ..., results['PM']['distances'], '基础模型 (PM) 的锚点视图')
    plot_full_anchor_view(axes[1], results['PM-SCL']['tsne_img'], ..., results['PM-SCL']['distances'], '对比学习模型 (PM-SCL) 的锚点视图')
    
    # ... (后续保存和显示代码保持不变) ...