import torch
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
# 注意：我们将不再使用sklearn的cosine_similarity，而是用PyTorch的实现
# from sklearn.metrics.pairwise import cosine_similarity 

# 导入项目模块
from model import AirQualityModel
from dataset import AirQualityDataset
from configs import experiment_configs

def get_embeddings_and_labels(model, data_loader, config, device, num_batches=5):
    """
    提取多个批次的特征嵌入和标签。
    """
    model.eval()
    model.training = True  # 强制模型返回嵌入
    config["use_contrastive"] = True

    all_img_embeddings = []
    all_num_embeddings = []
    all_labels = []

    with torch.no_grad():
        for i, (imgs, pollution, weather, adj, target) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            imgs, pollution, weather = imgs.to(device), pollution.to(device), weather.to(device)
            adj_hybrid, adj_phys = adj
            adj_mask = (adj_hybrid.to(device), adj_phys.to(device))
            
            _, img_embed, num_embed = model(imgs, pollution, weather, adj_mask=adj_mask)

            # --- 优化点：直接将GPU上的Tensor存起来，避免过早转到CPU ---
            all_img_embeddings.append(img_embed)
            all_num_embeddings.append(num_embed)
            all_labels.append(target.cpu()) # 标签可以直接转到CPU

    model.training = False
    
    # 在函数末尾一次性拼接并转移到CPU
    img_embeddings_gpu = torch.cat(all_img_embeddings)
    num_embeddings_gpu = torch.cat(all_num_embeddings)
    labels_np = torch.cat(all_labels).numpy().flatten()
    
    return img_embeddings_gpu, num_embeddings_gpu, labels_np


def calculate_similarities_batched(img_embeds_gpu, num_embeds_gpu, labels, 
                                   positive_threshold, negative_threshold_factor=5.0, 
                                   batch_size=32):
    """
    【优化版函数】
    根据标签阈值，分块计算正负样本对的跨模态余弦相似度，以节省内存。
    所有计算都在GPU上完成以提高速度。
    """
    n_samples = len(labels)
    device = img_embeds_gpu.device
    
    positive_sims = []
    negative_sims = []

    # 将标签转换为Tensor并放到GPU上，用于高效索引
    labels_gpu = torch.from_numpy(labels).to(device)

    # 对图像嵌入进行L2归一化
    img_embeds_gpu = torch.nn.functional.normalize(img_embeds_gpu, p=2, dim=1)

    for i in range(0, n_samples, batch_size):
        # --- 分块处理时空嵌入 ---
        start_idx = i
        end_idx = min(i + batch_size, n_samples)
        
        # 当前块的时空嵌入和标签
        num_embeds_batch = num_embeds_gpu[start_idx:end_idx]
        labels_batch = labels_gpu[start_idx:end_idx]
        
        # L2归一化
        num_embeds_batch = torch.nn.functional.normalize(num_embeds_batch, p=2, dim=1)
        
        # 计算当前块与所有图像嵌入的相似度矩阵
        # sim_matrix_batch 的形状是 [N_samples, batch_size]
        sim_matrix_batch = torch.matmul(img_embeds_gpu, num_embeds_batch.T)
        
        # 计算当前块与所有样本的标签差异
        # label_diff_matrix 的形状是 [N_samples, batch_size]
        label_diff_matrix = torch.abs(labels_gpu.unsqueeze(1) - labels_batch.unsqueeze(0))
        
        # 创建一个掩码来排除对角线元素（即样本自身与自身的比较）
        # identity_mask 的形状是 [N_samples, batch_size]
        identity_mask = torch.ones_like(label_diff_matrix, dtype=torch.bool)
        if end_idx > start_idx:
             # arange的范围需要匹配当前块在整个数据集中的索引
            batch_indices = torch.arange(start_idx, end_idx, device=device)
            # 将对应位置设置为False
            identity_mask[batch_indices, torch.arange(len(batch_indices), device=device)] = False

        # --- 使用掩码高效地筛选正负样本 ---
        # 正样本掩码
        pos_mask = (label_diff_matrix < positive_threshold) & identity_mask
        # 负样本掩码
        neg_mask = (label_diff_matrix > positive_threshold * negative_threshold_factor) & identity_mask
        
        # 提取相似度并转移到CPU
        positive_sims.extend(sim_matrix_batch[pos_mask].cpu().numpy().tolist())
        negative_sims.extend(sim_matrix_batch[neg_mask].cpu().numpy().tolist())

        # （可选）清理缓存，在内存极度紧张时有用
        # torch.cuda.empty_cache()
                
    return positive_sims, negative_sims


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="可视化对比学习对特征相似度分布的影响")
    # --- 用户需要修改这里的路径 ---
    parser.add_argument('--pm_scl_model_path', type=str, 
                        default='experiments/PM_SCL/best_model.pth',
                        help='已训练的 PM-SCL 模型权重路径')
    parser.add_argument('--pm_model_path', type=str, 
                        default='experiments/PM/best_model.pth',
                        help='已训练的基础版 PM 模型权重路径')
    # --------------------------
    parser.add_argument('--batch_size', type=int, default=64, help='数据加载器的批大小')
    parser.add_argument('--num_batches', type=int, default=5, help='用于可视化的批次数')
    parser.add_argument('--positive_threshold', type=float, default=1.0, 
                        help='定义正样本对的标签差异阈值, 应与训练时保持一致')
    parser.add_argument('--calc_batch_size', type=int, default=32, 
                        help='相似度计算时的内部批大小，用于控制内存')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载数据
    base_config = experiment_configs['PM']
    dataset = AirQualityDataset(base_config["pkl_file"], base_config["distances"], base_config)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. 加载模型
    print("正在加载模型...")
    config_scl = experiment_configs['PM_SCL']
    model_scl = AirQualityModel(config_scl).to(device)
    model_scl.load_state_dict(torch.load(args.pm_scl_model_path, map_location=device))

    config_pm = experiment_configs['PM']
    model_pm = AirQualityModel(config_pm).to(device)
    model_pm.load_state_dict(torch.load(args.pm_model_path, map_location=device))
    
    # 3. 提取嵌入 (注意，现在返回的是GPU上的Tensor)
    print("正在提取特征嵌入...")
    img_pm_gpu, num_pm_gpu, labels = get_embeddings_and_labels(model_pm, data_loader, config_pm, device, args.num_batches)
    img_scl_gpu, num_scl_gpu, _ = get_embeddings_and_labels(model_scl, data_loader, config_scl, device, args.num_batches)

    # 4. 计算相似度 (使用优化后的函数)
    print("正在计算正负样本对的相似度...")
    pos_sims_pm, neg_sims_pm = calculate_similarities_batched(
        img_pm_gpu, num_pm_gpu, labels, args.positive_threshold, batch_size=args.calc_batch_size
    )
    pos_sims_scl, neg_sims_scl = calculate_similarities_batched(
        img_scl_gpu, num_scl_gpu, labels, args.positive_threshold, batch_size=args.calc_batch_size
    )
    
    # 5. 创建DataFrame以便绘图 (后续代码不变)
    data = []
    for sim in pos_sims_pm:
        data.append({'模型': '基础模型 (PM)', '样本对类型': '正样本对', '余弦相似度': sim})
    for sim in neg_sims_pm:
        data.append({'模型': '基础模型 (PM)', '样本对类型': '负样本对', '余弦相似度': sim})
    for sim in pos_sims_scl:
        data.append({'模型': '对比学习模型 (PM-SCL)', '样本对类型': '正样本对', '余弦相似度': sim})
    for sim in neg_sims_scl:
        data.append({'模型': '对比学习模型 (PM-SCL)', '样本对类型': '负样本对', '余弦相似度': sim})
    
    df = pd.DataFrame(data)

    # 6. 可视化
    print("正在生成KDE图...")
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    g = sns.displot(
        df, x="余弦相似度", col="模型", hue="样本对类型",
        kind="kde", fill=True, common_norm=False,
        palette=sns.color_palette('bright')[:2],
        height=5, aspect=1.2,
    )
    
    g.fig.suptitle('对比学习对跨模态特征相似度分布的影响', fontsize=16, y=1.03)
    g.set_axis_labels("跨模态余弦相似度", "密度")
    g.set_titles("模型: {col_name}")

    save_path = 'similarity_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {save_path}")
    
    plt.show()