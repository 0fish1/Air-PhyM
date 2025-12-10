import torch
import numpy as np
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# 确保可以从项目目录中导入所需的模块
from model import AirQualityModel
from dataset import AirQualityDataset
from configs import experiment_configs

def get_embeddings(model, data_loader, config, device):
    """
    从给定的模型和数据加载器中提取特征嵌入。

    Args:
        model (nn.Module): 已经加载了权重的模型。
        data_loader (DataLoader): 测试数据加载器。
        config (dict): 模型的配置字典。
        device (torch.device): 运行设备 (cuda/cpu)。

    Returns:
        tuple: (图像嵌入, 时空嵌入, 真实标签) 的Numpy数组。
    """
    model.eval()
    
    # 这是一个小技巧：为了让模型返回中间特征，我们临时将模式设置为训练状态
    # 但在 torch.no_grad() 环境下运行，所以不会计算梯度或更新权重。
    model.training = True
    config["use_contrastive"] = True # 强制模型返回多模态嵌入

    img_embeddings = []
    num_embeddings = []
    labels = []

    with torch.no_grad():
        # 我们只需要一批数据进行可视化
        imgs, pollution, weather, adj, target = next(iter(data_loader))
        
        imgs = imgs.to(device)
        pollution = pollution.to(device)
        weather = weather.to(device)
        adj_hybrid, adj_phys = adj
        adj_mask = (adj_hybrid.to(device), adj_phys.to(device))
        
        # 模型在前向传播时会返回预测值、图像嵌入和时空嵌入
        _, img_embed, num_embed = model(imgs, pollution, weather, adj_mask=adj_mask)

        img_embeddings.append(img_embed.cpu().numpy())
        num_embeddings.append(num_embed.cpu().numpy())
        labels.append(target.cpu().numpy())

    # 恢复模型的评估模式
    model.training = False

    # 将列表转换为Numpy数组
    img_embeddings = np.concatenate(img_embeddings, axis=0)
    num_embeddings = np.concatenate(num_embeddings, axis=0)
    labels = np.concatenate(labels, axis=0).flatten()
    
    return img_embeddings, num_embeddings, labels

def plot_tsne(ax, tsne_img, tsne_num, labels, title):
    """
    在指定的坐标轴上绘制t-SNE可视化图。

    Args:
        ax (matplotlib.axes.Axes): Matplotlib的子图对象。
        tsne_img (np.array): 降维后的图像特征 (N, 2)。
        tsne_num (np.array): 降维后的时空特征 (N, 2)。
        labels (np.array): 真实标签 (N,)。
        title (str): 子图的标题。
    """
    # 绘制视觉特征（圆圈）
    scatter_img = ax.scatter(
        tsne_img[:, 0], tsne_img[:, 1],
        c=labels,
        cmap='viridis',
        marker='o',
        alpha=0.8,
        label='视觉特征 (Image)'
    )
    
    # 绘制时空特征（叉号）
    ax.scatter(
        tsne_num[:, 0], tsne_num[:, 1],
        c=labels,
        cmap='viridis',
        marker='x',
        s=50, # 稍微调大一点以便看清
        alpha=0.8,
        label='时空特征 (Spatio-temporal)'
    )
    
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    
    # 添加颜色条
    cbar = plt.colorbar(scatter_img, ax=ax)
    cbar.set_label('真实 PM2.5 浓度值')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="可视化模型的多模态特征嵌入空间")
    # --- 用户需要修改这里的路径 ---
    parser.add_argument('--pm_scl_model_path', type=str, 
                        default='experiments/PM_SCL/best_model.pth',
                        help='已训练的 PM-SCL 模型权重路径')
    parser.add_argument('--pm_model_path', type=str, 
                        default='experiments/PM/best_model.pth',
                        help='已训练的基础版 PM 模型权重路径')
    # --------------------------
    parser.add_argument('--batch_size', type=int, default=64, help='用于可视化的样本数量')
    
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载数据
    # 我们使用基础配置来加载数据集
    base_config = experiment_configs['PM']
    dataset = AirQualityDataset(base_config["pkl_file"], base_config["distances"], base_config)
    
    # 创建一个不打乱的数据加载器，以保证两个模型使用完全相同的样本
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. 加载模型
    print("正在加载模型...")
    # 加载 PM-SCL 模型
    config_scl = experiment_configs['PM_SCL']
    model_scl = AirQualityModel(config_scl).to(device)
    model_scl.load_state_dict(torch.load(args.pm_scl_model_path, map_location=device))

    # 加载基础版 PM 模型
    config_pm = experiment_configs['PM']
    model_pm = AirQualityModel(config_pm).to(device)
    model_pm.load_state_dict(torch.load(args.pm_model_path, map_location=device))
    
    # 3. 提取嵌入
    print("正在为 PM (基础版) 模型提取特征嵌入...")
    img_embed_pm, num_embed_pm, labels_pm = get_embeddings(model_pm, data_loader, config_pm, device)
    
    print("正在为 PM-SCL (对比学习版) 模型提取特征嵌入...")
    img_embed_scl, num_embed_scl, labels_scl = get_embeddings(model_scl, data_loader, config_scl, device)

    # 4. t-SNE 降维
    print("正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='pca', learning_rate='auto')
    
    # 对 PM 模型嵌入进行降维
    all_embed_pm = np.vstack((img_embed_pm, num_embed_pm))
    tsne_results_pm = tsne.fit_transform(all_embed_pm)
    tsne_img_pm, tsne_num_pm = np.split(tsne_results_pm, 2)
    
    # 对 PM-SCL 模型嵌入进行降维
    all_embed_scl = np.vstack((img_embed_scl, num_embed_scl))
    tsne_results_scl = tsne.fit_transform(all_embed_scl)
    tsne_img_scl, tsne_num_scl = np.split(tsne_results_scl, 2)
    
    # 5. 可视化
    print("正在生成可视化图表...")
    sns.set_style("whitegrid")
    # 设置中文字体，请根据您的系统配置选择合适的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # 绘制基础版 PM 模型的可视化
    plot_tsne(axes[0], tsne_img_pm, tsne_num_pm, labels_pm, '基础模型 (PM) 特征空间')
    
    # 绘制 PM-SCL 模型的可视化
    plot_tsne(axes[1], tsne_img_scl, tsne_num_scl, labels_scl, '对比学习模型 (PM-SCL) 特征空间')
    
    fig.suptitle('多模态特征嵌入空间 t-SNE 可视化对比', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存图像
    save_path = 'embedding_visualization.png'
    plt.savefig(save_path, dpi=300)
    print(f"可视化结果已保存至: {save_path}")
    
    plt.show()