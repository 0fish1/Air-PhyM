import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import os

# 导入您项目中的模块
# 确保此脚本与 model.py, dataset.py, configs.py 在同一目录下
from model import AirQualityModel
from dataset import AirQualityDataset
from configs import experiment_configs

def get_all_embeddings(model, data_loader, config, device):
    """
    为批次中的所有数据提取两种模态的特征嵌入和真实标签。
    """
    model.eval()
    # 强制模型进入返回多模态嵌入的模式
    model.training = True
    config["use_contrastive"] = True

    with torch.no_grad():
        # 从数据加载器中获取一个批次的数据
        imgs, pollution, weather, adj, target = next(iter(data_loader))
        
        # 将数据移动到指定设备
        imgs = imgs.to(device)
        pollution = pollution.to(device)
        weather = weather.to(device)
        adj_hybrid, adj_phys = adj
        adj_mask = (adj_hybrid.to(device), adj_phys.to(device))
        
        # 模型前向传播，获取嵌入
        _, img_embed, num_embed = model(imgs, pollution, weather, adj_mask=adj_mask)

    # 恢复模型的评估模式
    model.training = False
    
    # 将结果从GPU转移到CPU并转换为Numpy数组
    return (img_embed.cpu().numpy(), 
            num_embed.cpu().numpy(), 
            target.cpu().numpy().flatten())

def plot_final_anchor_view(ax, tsne_img, tsne_num, labels, anchor_info, title):
    """
    绘制最终的、最完整的锚点视图，包含“自我对齐”和“泛化对齐”。
    """
    anchor_idx = anchor_info['anchor_idx']
    distances = anchor_info['distances']

    # 1. 绘制所有点的暗色背景
    ax.scatter(tsne_img[:, 0], tsne_img[:, 1], c='gray', alpha=0.1, marker='o', label='_nolegend_')
    ax.scatter(tsne_num[:, 0], tsne_num[:, 1], c='gray', alpha=0.1, marker='x', label='_nolegend_')

    # 2. 突出显示锚点和各类邻居
    anchor_pos = tsne_img[anchor_idx]
    anchor_label = labels[anchor_idx]
    
    # 绘制锚点 (星形)
    ax.scatter(anchor_pos[0], anchor_pos[1], c=[anchor_label], cmap='viridis',
               marker='*', s=400, edgecolor='black', zorder=5, 
               label=f'锚点 (图像, PM2.5={anchor_label:.1f})')

    # --- 绘制最关键的“自我对齐”正样本 ---
    self_numerical_pos = tsne_num[anchor_idx]
    self_dist = distances['self_alignment']
    # 用五角星标记自身时空特征
    ax.scatter(self_numerical_pos[0], self_numerical_pos[1], c=[anchor_label], cmap='viridis',
               marker='p', s=250, edgecolor='gold', linewidth=2, zorder=7,
               label=f'自我对齐正样本 (d={self_dist:.2f})')
    # 绘制连接线
    ax.plot([anchor_pos[0], self_numerical_pos[0]], [anchor_pos[1], self_numerical_pos[1]], 
            color='gold', linestyle='-', linewidth=3, zorder=6)

    # --- 绘制泛化的正样本邻居 ---
    # 跨模态正样本 (圆形)
    indices = anchor_info['cross_modal_pos']
    if len(indices) > 0:
        points = tsne_num[indices]
        ax.scatter(points[:, 0], points[:, 1], c=labels[indices], cmap='viridis',
                   marker='o', s=150, edgecolor='green', linewidth=2, zorder=4, label='泛化正样本 (跨模态)')
        for idx in indices:
            p_pos = tsne_num[idx]
            dist = distances['cross_modal'][idx]
            ax.plot([anchor_pos[0], p_pos[0]], [anchor_pos[1], p_pos[1]], 'g-', alpha=0.7)
            ax.text((anchor_pos[0]+p_pos[0])/2, (anchor_pos[1]+p_pos[1])/2, f'{dist:.2f}', color='darkgreen', fontsize=9, bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))

    # 同模态正样本 (方形)
    indices = anchor_info['intra_modal_pos']
    if len(indices) > 0:
        points = tsne_img[indices]
        ax.scatter(points[:, 0], points[:, 1], c=labels[indices], cmap='viridis',
                   marker='s', s=150, edgecolor='blue', linewidth=2, zorder=4, label='泛化正样本 (同模态)')
        for idx in indices:
            p_pos = tsne_img[idx]
            dist = distances['intra_modal'][idx]
            ax.plot([anchor_pos[0], p_pos[0]], [anchor_pos[1], p_pos[1]], 'b:', alpha=0.7, linewidth=2)
            ax.text((anchor_pos[0]+p_pos[0])/2, (anchor_pos[1]+p_pos[1])/2, f'{dist:.2f}', color='darkblue', fontsize=9, bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))

    # --- 绘制负样本 ---
    indices = anchor_info['negative_indices']
    if len(indices) > 0:
        points = tsne_num[indices]
        ax.scatter(points[:, 0], points[:, 1], c=labels[indices], cmap='viridis',
                   marker='X', s=150, edgecolor='red', linewidth=2, zorder=4, label='负样本 (跨模态)')
        for idx in indices:
            n_pos = tsne_num[idx]
            dist = distances['cross_modal'][idx]
            ax.plot([anchor_pos[0], n_pos[0]], [anchor_pos[1], n_pos[1]], 'r--', alpha=0.6)
            ax.text((anchor_pos[0]+n_pos[0])/2, (anchor_pos[1]+n_pos[1])/2, f'{dist:.2f}', color='darkred', fontsize=9, bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))
    
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == '__main__':
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="为论文生成最终的、最完整的锚点视图对比图")
    parser.add_argument('--pm_scl_model_path', type=str, default='experiments/PM_SCL/best_model.pth')
    parser.add_argument('--pm_model_path', type=str, default='experiments/PM/best_model.pth')
    parser.add_argument('--batch_size', type=int, default=256, help='用于可视化的样本总数')
    parser.add_argument('--anchor_idx', type=int, default=3, help='锚点样本在批次中的索引')
    parser.add_argument('--num_positives', type=int, default=2, help='显示最近的泛化正样本数量（同/跨模态各2个）')
    parser.add_argument('--num_negatives', type=int, default=3, help='显示最远的负样本数量')
    parser.add_argument('--positive_threshold', type=float, default=1.0, help='正样本标签差异阈值')
    args = parser.parse_args()

    # --- 设备选择 ---
    device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 1. 加载数据
    print("正在加载数据...")
    base_config = experiment_configs['PM']
    dataset = AirQualityDataset(base_config["pkl_file"], base_config["distances"], base_config)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. 加载模型
    print("正在加载模型...")
    config_scl = experiment_configs['PM_SCL']
    model_scl = AirQualityModel(config_scl).to(device)
    model_scl.load_state_dict(torch.load(args.pm_scl_model_path, map_location=device))
    config_pm = experiment_configs['PM']
    model_pm = AirQualityModel(config_pm).to(device)
    model_pm.load_state_dict(torch.load(args.pm_model_path, map_location=device))

    # 3. 提取嵌入
    print("正在提取特征嵌入...")
    img_embeds_pm, num_embeds_pm, labels = get_all_embeddings(model_pm, data_loader, config_pm, device)
    img_embeds_scl, num_embeds_scl, _ = get_all_embeddings(model_scl, data_loader, config_scl, device)

    # 4. 核心分析逻辑
    analysis_results = {}
    for model_name, (img_embeds, num_embeds) in {"PM": (img_embeds_pm, num_embeds_pm), "PM-SCL": (img_embeds_scl, num_embeds_scl)}.items():
        print(f"正在分析模型: {model_name}")
        
        img_embeds_norm = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
        num_embeds_norm = num_embeds / np.linalg.norm(num_embeds, axis=1, keepdims=True)
        
        # t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(np.vstack([img_embeds, num_embeds]))
        tsne_img, tsne_num = np.split(tsne_results, 2)

        # --- 锚点分析逻辑重构 ---
        anchor_embed = img_embeds_norm[args.anchor_idx:args.anchor_idx+1]
        anchor_label = labels[args.anchor_idx]
        
        cross_modal_distances = cdist(anchor_embed, num_embeds_norm, 'euclidean').flatten()
        intra_modal_distances = cdist(anchor_embed, img_embeds_norm, 'euclidean').flatten()
        
        # 1. 直接获取“自我对齐”的距离
        self_alignment_dist = cross_modal_distances[args.anchor_idx]
        
        # 2. 在“其他”样本中寻找泛化邻居
        candidate_indices = np.arange(len(labels))[np.arange(len(labels)) != args.anchor_idx]
        label_diffs = np.abs(labels[candidate_indices] - anchor_label)
        
        possible_pos_indices = candidate_indices[label_diffs < args.positive_threshold]
        possible_neg_indices = candidate_indices[label_diffs >= args.positive_threshold]
        
        top_k_cross_positives = possible_pos_indices[np.argsort(cross_modal_distances[possible_pos_indices])][:args.num_positives]
        top_k_intra_positives = possible_pos_indices[np.argsort(intra_modal_distances[possible_pos_indices])][:args.num_positives]
        top_m_negatives = possible_neg_indices[np.argsort(cross_modal_distances[possible_neg_indices])][-args.num_negatives:]
        
        analysis_results[model_name] = {
            "tsne_img": tsne_img, "tsne_num": tsne_num,
            "anchor_info": {
                "anchor_idx": args.anchor_idx,
                "self_alignment_dist": self_alignment_dist,
                "cross_modal_pos": top_k_cross_positives,
                "intra_modal_pos": top_k_intra_positives,
                "negative_indices": top_m_negatives,
                "distances": {"cross_modal": cross_modal_distances, "intra_modal": intra_modal_distances, "self_alignment": self_alignment_dist}
            }
        }

    # 5. 可视化
    print("正在生成最终的锚点视图对比图...")
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    
    plot_final_anchor_view(axes[0], analysis_results['PM']['tsne_img'], analysis_results['PM']['tsne_num'], labels,
                           analysis_results['PM']['anchor_info'], '基础模型 (PM) 的锚点视图')
                     
    plot_final_anchor_view(axes[1], analysis_results['PM-SCL']['tsne_img'], analysis_results['PM-SCL']['tsne_num'], labels,
                           analysis_results['PM-SCL']['anchor_info'], 'PM-SCL模型 的锚点视图')
    
    fig.suptitle(f'锚点样本 (索引 {args.anchor_idx}) 与其邻居的特征空间距离对比', fontsize=18, y=0.96)
    
    # --- 建议：创建专门的输出文件夹 ---
    output_dir = 'final_anchor_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'final_anchor_view_idx{args.anchor_idx}.png')
    # ------------------------------------
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"最终可视化结果已保存至: {save_path}")
    
    plt.show()