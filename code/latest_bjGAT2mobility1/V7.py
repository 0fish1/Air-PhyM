import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import os
import random

# 导入您项目中的模块
# 确保此脚本与 model.py, dataset.py, configs.py 在同一目录下
from model import AirQualityModel
from dataset import AirQualityDataset
from configs import experiment_configs

# =============================================================================
# 1. 辅助函数
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_all_embeddings(model, data_loader, config, device):
    """
    为批次中的所有数据提取两种模态的特征嵌入和真实标签。
    """
    model.eval()
    # 强制模型进入返回多模态嵌入的模式
    # 这是一个小技巧，让模型即使在eval模式下也执行返回嵌入的分支
    original_training_state = model.training
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

    # 恢复模型的原始状态
    model.training = original_training_state
    
    # 将结果从GPU转移到CPU并转换为Numpy数组
    return (img_embed.cpu().numpy(), 
            num_embed.cpu().numpy(), 
            target.cpu().numpy().flatten())

# =============================================================================
# 2. 主程序
# =============================================================================
if __name__ == '__main__':
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="生成出版级的特征空间可视化图")
    parser.add_argument('--scl_exp_name', type=str, default='PM_SCL_gated', help='包含SCL的实验配置名称')
    parser.add_argument('--base_exp_name', type=str, default='no_SCL', help='不包含SCL的基线实验配置名称')
    parser.add_argument('--batch_size', type=int, default=256, help='用于可视化的样本总数')
    parser.add_argument('--anchor_idx', type=int, default=3, help='锚点样本在批次中的索引')
    parser.add_argument('--positive_threshold', type=float, default=15.0, help='正样本标签差异阈值δ')
    args = parser.parse_args()

    # --- 初始化 ---
    config = experiment_configs[args.scl_exp_name]
    set_seed(config['seed'])
    device = torch.device(config['device'])
    print(f"使用设备: {device}")

    # --- 加载数据 ---
    print("正在加载数据...")
    dataset = AirQualityDataset(config["pkl_file"], config["distances"], config)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- 加载模型 ---
    print("正在加载模型...")
    scl_model_path = f"experiments/{args.scl_exp_name}/best_model.pth"
    base_model_path = f"experiments/{args.base_exp_name}/best_model.pth"
    if not os.path.exists(scl_model_path) or not os.path.exists(base_model_path):
        print("错误: 找不到一个或两个模型文件。请确保您已成功训练以下两个模型:")
        print(f"  - {scl_model_path}")
        print(f"  - {base_model_path}")
        exit()
        
    model_scl = AirQualityModel(config).to(device)
    model_scl.load_state_dict(torch.load(scl_model_path, map_location=device))
    
    config_base = experiment_configs[args.base_exp_name]
    model_base = AirQualityModel(config_base).to(device)
    model_base.load_state_dict(torch.load(base_model_path, map_location=device))

    # --- 提取所有嵌入 ---
    print("正在提取特征嵌入...")
    img_embeds_base, num_embeds_base, labels = get_all_embeddings(model_base, data_loader, config_base, device)
    img_embeds_scl, num_embeds_scl, _ = get_all_embeddings(model_scl, data_loader, config, device)

    # --- 准备绘图所需的数据字典 ---
    print("正在准备绘图数据...")
    plot_data = {}
    models_to_process = {
        'Baseline Model (without SCL)': (img_embeds_base, num_embeds_base),
        'Proposed PM-SCL Model': (img_embeds_scl, num_embeds_scl)
    }

    for model_name, (img_embeds, num_embeds) in models_to_process.items():
        # L2 归一化
        img_embeds_norm = img_embeds / (np.linalg.norm(img_embeds, axis=1, keepdims=True) + 1e-9)
        num_embeds_norm = num_embeds / (np.linalg.norm(num_embeds, axis=1, keepdims=True) + 1e-9)
        
        # t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=30, random_state=config['seed'], init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(np.vstack([img_embeds_norm, num_embeds_norm]))
        tsne_img, tsne_num = np.split(tsne_results, 2)

        # 寻找关键邻居
        anchor_embed = img_embeds_norm[args.anchor_idx:args.anchor_idx+1]
        anchor_label = labels[args.anchor_idx]
        
        cross_modal_distances = cdist(anchor_embed, num_embeds_norm, 'euclidean').flatten()
        intra_modal_distances = cdist(anchor_embed, img_embeds_norm, 'euclidean').flatten()
        
        candidate_indices = np.arange(len(labels))
        is_not_anchor = candidate_indices != args.anchor_idx
        label_diffs = np.abs(labels - anchor_label)
        
        pos_indices = candidate_indices[is_not_anchor & (label_diffs < args.positive_threshold)]
        neg_indices = candidate_indices[is_not_anchor & (label_diffs >= args.positive_threshold)]
        
        key_indices = {
            'paired_positive': args.anchor_idx,
            'cross_modal_positive': pos_indices[np.argmin(cross_modal_distances[pos_indices])] if len(pos_indices) > 0 else -1,
            'intra_modal_positive': pos_indices[np.argmin(intra_modal_distances[pos_indices])] if len(pos_indices) > 0 else -1,
            'cross_modal_negative': neg_indices[np.argmin(cross_modal_distances[neg_indices])] if len(neg_indices) > 0 else -1,
            'intra_modal_negative': neg_indices[np.argmin(intra_modal_distances[neg_indices])] if len(neg_indices) > 0 else -1,
        }
        
        plot_data[model_name] = {
            'tsne_img': tsne_img, 'tsne_num': tsne_num,
            'coords': { k: (tsne_img[v] if 'intra_modal' in k else tsne_num[v]) if k != 'anchor' else tsne_img[args.anchor_idx] for k, v in key_indices.items() if v != -1 },
            'distances': {
                'paired_positive': cross_modal_distances[key_indices['paired_positive']],
                'cross_modal_positive': cross_modal_distances[key_indices['cross_modal_positive']],
                'intra_modal_positive': intra_modal_distances[key_indices['intra_modal_positive']],
                'cross_modal_negative': cross_modal_distances[key_indices['cross_modal_negative']],
                'intra_modal_negative': intra_modal_distances[key_indices['intra_modal_negative']],
            }
        }

    # --- 绘图 ---
    print("正在生成最终可视化图...")
    fig, axes = plt.subplots(1, 2, figsize=(28, 13))
    plt.style.use('seaborn-v0_8-whitegrid')

    styles = {
        'anchor': {'marker': '*', 's': 1000, 'c': '#4B0082', 'edgecolors': 'black', 'zorder': 10, 'label': f'Anchor (Image, PM2.5={labels[args.anchor_idx]:.1f})'},
        'paired_positive': {'marker': 'p', 's': 500, 'c': '#FFD700', 'edgecolors': 'black', 'zorder': 9, 'linestyle': 'solid', 'linewidth': 4},
        'intra_modal_positive': {'marker': 's', 's': 500, 'c': '#00008B', 'edgecolors': 'black', 'zorder': 9, 'linestyle': (0, (5, 5)), 'linewidth': 3.5},
        'cross_modal_positive': {'marker': 'o', 's': 500, 'c': '#2E8B57', 'edgecolors': 'black', 'zorder': 9, 'linestyle': 'solid', 'linewidth': 3.5},
        'intra_modal_negative': {'marker': 'X', 's': 500, 'c': '#696969', 'edgecolors': 'black', 'zorder': 9, 'linestyle': (0, (3, 3)), 'linewidth': 4},
        'cross_modal_negative': {'marker': 'X', 's': 500, 'c': '#B22222', 'edgecolors': 'black', 'zorder': 9, 'linestyle': (0, (5, 2)), 'linewidth': 4}
    }
    
    legend_labels = {
        'paired_positive': 'Paired Positive (Spatio-temporal)',
        'intra_modal_positive': 'Intra-modal Positive (Image)',
        'cross_modal_positive': 'Cross-modal Positive (Spatio-temporal)',
        'intra_modal_negative': 'Intra-modal Negative (Image)',
        'cross_modal_negative': 'Cross-modal Negative (Spatio-temporal)'
    }

    for i, model_name in enumerate(models_to_process.keys()):
        ax = axes[i]
        data = plot_data[model_name]
        coords, distances = data['coords'], data['distances']
        
        ax.scatter(data['tsne_img'][:, 0], data['tsne_img'][:, 1], c='#E5E5E5', marker='x', alpha=0.7, s=50, label='Other Image Samples')
        ax.scatter(data['tsne_num'][:, 0], data['tsne_num'][:, 1], c='#F5F5F5', marker='o', alpha=0.6, s=50, label='Other Spatio-Temporal Samples')

        anchor_pos = coords['anchor']
        ax.scatter(anchor_pos[0], anchor_pos[1], **styles['anchor'])

        for key in legend_labels.keys():
            if key not in coords: continue
            pos = coords[key]
            style = styles[key]
            dist = distances[key]
            
            ax.scatter(pos[0], pos[1], marker=style['marker'], s=style['s'], c=style['c'], edgecolors=style['edgecolors'], zorder=style['zorder'], label=legend_labels[key])
            
            line_x, line_y = [anchor_pos[0], pos[0]], [anchor_pos[1], pos[1]]
            ax.plot(line_x, line_y, color=style['c'], linestyle=style['linestyle'], linewidth=style.get('linewidth', 2), zorder=8)
            
            mid_point = np.array([(anchor_pos[0] + pos[0]) / 2, (anchor_pos[1] + pos[1]) / 2])
            direction_vec = pos - anchor_pos
            perp_vec = np.array([-direction_vec[1], direction_vec[0]])
            norm_perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-9)
            
            offset_scale = 0.1 * np.sqrt(np.sum(direction_vec**2))
            label_pos = mid_point + norm_perp_vec * offset_scale
            
            ax.text(label_pos[0], label_pos[1], f'{dist:.2f}', fontsize=18, fontweight='bold', color=style['c'], ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))

        ax.set_title(model_name, fontsize=28, pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('t-SNE Dimension 1', fontsize=22)
        if i == 0:
            ax.set_ylabel('t-SNE Dimension 2', fontsize=22)
        
        handles, labels_legend = ax.get_legend_handles_labels()
        order = [2, 3, 4, 5, 6, 7, 0, 1] 
        ax.legend([handles[idx] for idx in order], [labels_legend[idx] for idx in order], loc='upper left', fontsize=16, frameon=True, shadow=True, title='Sample Types', title_fontsize='18')

    fig.suptitle('Effect of Supervised Contrastive Learning on the Embedding Space', fontsize=40, y=0.98)
    fig.text(0.5, -0.01, 
             'Figure X: A t-SNE visualization of the embedding space, comparing the Baseline model (left) with the proposed PM-SCL model (right) from the perspective of a single anchor sample.\n'
             'Distances (d) represent the Euclidean Distance in the high-dimensional, L2-normalized projection space. The PM-SCL model demonstrates effective representation alignment by\n'
             'restructuring the space to pull positive samples closer and push negative samples further away, while also merging the distributions of the two modalities (grey points).',
             ha='center', fontsize=18, style='italic', wrap=True, linespacing=1.5)

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    output_dir = f"experiments/{args.scl_exp_name}/"
    save_path = os.path.join(output_dir, "embedding_space_visualization.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Final visualization from real data saved to {save_path}")
    plt.show()