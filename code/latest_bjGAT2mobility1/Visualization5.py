# import torch
# import numpy as np
# import argparse
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import DataLoader
# from sklearn.metrics.pairwise import cosine_similarity

# # 导入项目模块
# from model import AirQualityModel
# from dataset import AirQualityDataset
# from configs import experiment_configs

# def get_all_embeddings(model, data_loader, config, device):
#     """为一批数据提取所有模态的嵌入和标签。"""
#     model.eval()
#     model.training = True
#     config["use_contrastive"] = True
#     with torch.no_grad():
#         imgs, pollution, weather, adj, target = next(iter(data_loader))
#         imgs, pollution, weather = imgs.to(device), pollution.to(device), weather.to(device)
#         adj_hybrid, adj_phys = adj
#         adj_mask = (adj_hybrid.to(device), adj_phys.to(device))
#         _, img_embed, num_embed = model(imgs, pollution, weather, adj_mask=adj_mask)
#     model.training = False
#     return (img_embed.cpu().numpy(), 
#             num_embed.cpu().numpy(), 
#             target.cpu().numpy().flatten())

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="为论文生成最终的、最合理的相似度矩阵对比图")
#     parser.add_argument('--pm_scl_model_path', type=str, default='experiments/PM_SCL/best_model.pth')
#     parser.add_argument('--pm_model_path', type=str, default='experiments/PM/best_model.pth')
#     parser.add_argument('--batch_size', type=int, default=128, help='用于可视化的样本总数，建议值64-256')
#     args = parser.parse_args()

#     device = torch.device("cpu")
#     print(f"使用设备: {device}")

#     # 1. 加载数据
#     base_config = experiment_configs['PM']
#     dataset = AirQualityDataset(base_config["pkl_file"], base_config["distances"], base_config)
#     # 使用 shuffle=True 可以每次看到不同批次的效果，更能验证方法的普适性
#     # 如果要固定图片，请使用 shuffle=False
#     data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

#     # 2. 加载模型
#     print("正在加载模型...")
#     config_scl = experiment_configs['PM_SCL']
#     model_scl = AirQualityModel(config_scl).to(device)
#     model_scl.load_state_dict(torch.load(args.pm_scl_model_path, map_location=device))
#     config_pm = experiment_configs['PM']
#     model_pm = AirQualityModel(config_pm).to(device)
#     model_pm.load_state_dict(torch.load(args.pm_model_path, map_location=device))

#     # 3. 提取嵌入
#     print("正在提取特征嵌入...")
#     img_embeds_pm, num_embeds_pm, labels = get_all_embeddings(model_pm, data_loader, config_pm, device)
#     img_embeds_scl, num_embeds_scl, _ = get_all_embeddings(model_scl, data_loader, config_scl, device)

#     # 4. 核心分析逻辑：计算并排序相似度矩阵
#     # 按标签值对样本进行排序，获取排序索引
#     sort_indices = np.argsort(labels)
    
#     analysis_results = {}
#     for model_name, (img_embeds, num_embeds) in {"PM": (img_embeds_pm, num_embeds_pm), "PM-SCL": (img_embeds_scl, num_embeds_scl)}.items():
#         # 归一化特征
#         img_embeds_norm = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
#         num_embeds_norm = num_embeds / np.linalg.norm(num_embeds, axis=1, keepdims=True)
        
#         # 计算完整的跨模态相似度矩阵
#         similarity_matrix = cosine_similarity(img_embeds_norm, num_embeds_norm)
        
#         # 使用排序索引来重排矩阵的行和列
#         sorted_matrix = similarity_matrix[sort_indices, :][:, sort_indices]
#         analysis_results[model_name] = sorted_matrix

#     # 5. 可视化
#     print("正在生成最终的相似度矩阵热图...")
#     sns.set_style("white")
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False

#     fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
#     # 绘制基础模型的热图
#     im1 = axes[0].imshow(analysis_results['PM'], cmap='viridis', vmin=-0.1, vmax=0.2) # 调整vmin/vmax来统一色阶
#     axes[0].set_title('(a) 基础模型 (PM)', fontsize=16)
#     axes[0].set_xlabel('时空特征 (按PM2.5值排序)', fontsize=12)
#     axes[0].set_ylabel('视觉特征 (按PM2.5值排序)', fontsize=12)

#     # 绘制PM-SCL模型的热图
#     im2 = axes[1].imshow(analysis_results['PM-SCL'], cmap='viridis', vmin=-0.1, vmax=0.2)
#     axes[1].set_title('(b) PM-SCL模型', fontsize=16)
#     axes[1].set_xlabel('时空特征 (按PM2.5值排序)', fontsize=12)
#     axes[1].set_ylabel('') # 隐藏Y轴标签以保持简洁
#     axes[1].set_yticks([])

#     fig.suptitle('监督对比学习对跨模态特征相似度结构的影响', fontsize=20, y=0.98)
    
#     # 添加一个共享的颜色条
#     cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
#     cbar.set_label('跨模态余弦相似度', fontsize=12)
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     save_path = 'final_similarity_matrix.png'
#     plt.savefig(save_path, dpi=300)
#     print(f"最终可视化结果已保存至: {save_path}")
    
#     plt.show()









import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter # 导入高斯滤波器

# 导入您项目中的模块
# 确保此脚本与 model.py, dataset.py, configs.py 在同一目录下
from model import AirQualityModel
from dataset import AirQualityDataset
from configs import experiment_configs

def get_all_embeddings(model, data_loader, config, device):
    """
    为批次中的所有数据提取两种模态的特征嵌入和真实标签。
    
    Args:
        model (torch.nn.Module): 已加载权重的模型。
        data_loader (DataLoader): 数据加载器。
        config (dict): 实验配置。
        device (torch.device): 运行设备。

    Returns:
        tuple: (图像嵌入, 时空嵌入, 标签) 的Numpy数组。
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

if __name__ == '__main__':
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="为论文生成最终的、经过高斯平滑优化的相似度矩阵对比图")
    
    # 模型路径参数
    parser.add_argument('--pm_scl_model_path', type=str, 
                        default='experiments/PM_SCL/best_model.pth',
                        help='已训练的 PM-SCL 模型权重路径')
    parser.add_argument('--pm_model_path', type=str, 
                        default='experiments/PM/best_model.pth',
                        help='已训练的基础版 PM 模型权重路径')
    
    # 数据与可视化参数
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='用于可视化的样本总数。建议值在64到256之间，以获得稳定的结果。')
    
    # 优化参数
    parser.add_argument('--smoothing_sigma', type=float, default=1.0, 
                        help='高斯平滑的标准差。值越大，平滑效果越强。建议范围1.0-1.5。设置为0则不进行平滑。')
    
    args = parser.parse_args()

    # --- 设备选择 ---
    # 默认使用CPU，以确保在任何环境下都能稳定运行，避免CUDA内存问题
    device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 1. 加载数据
    print("正在加载数据...")
    base_config = experiment_configs['PM']
    dataset = AirQualityDataset(base_config["pkl_file"], base_config["distances"], base_config)
    # 使用 shuffle=False 确保每次运行加载的是数据集头部的相同批次，保证图片的可复现性
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. 加载模型
    print("正在加载模型...")
    # 加载 PM-SCL 模型
    config_scl = experiment_configs['PM_SCL']
    model_scl = AirQualityModel(config_scl).to(device)
    model_scl.load_state_dict(torch.load(args.pm_scl_model_path, map_location=device))
    # 加载基础 PM 模型
    config_pm = experiment_configs['PM']
    model_pm = AirQualityModel(config_pm).to(device)
    model_pm.load_state_dict(torch.load(args.pm_model_path, map_location=device))

    # 3. 提取所有需要的嵌入和标签
    print("正在提取特征嵌入...")
    img_embeds_pm, num_embeds_pm, labels = get_all_embeddings(model_pm, data_loader, config_pm, device)
    img_embeds_scl, num_embeds_scl, _ = get_all_embeddings(model_scl, data_loader, config_scl, device)

    # 4. 核心分析逻辑：计算、排序，并进行高斯平滑
    print("正在计算和处理相似度矩阵...")
    # 根据真实标签值获取排序索引
    sort_indices = np.argsort(labels)
    
    analysis_results = {}
    for model_name, (img_embeds, num_embeds) in {
        "PM": (img_embeds_pm, num_embeds_pm), 
        "PM-SCL": (img_embeds_scl, num_embeds_scl)
    }.items():
        # 归一化特征向量，以便余弦相似度计算
        img_embeds_norm = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
        num_embeds_norm = num_embeds / np.linalg.norm(num_embeds, axis=1, keepdims=True)
        
        # 计算完整的 N x N 跨模态余弦相似度矩阵
        similarity_matrix = cosine_similarity(img_embeds_norm, num_embeds_norm)
        
        # 使用排序索引来重排矩阵的行和列
        sorted_matrix = similarity_matrix[sort_indices, :][:, sort_indices]
        
        # 应用高斯平滑进行优化
        if args.smoothing_sigma > 0:
            smoothed_matrix = gaussian_filter(sorted_matrix, sigma=args.smoothing_sigma)
        else:
            smoothed_matrix = sorted_matrix
            
        analysis_results[model_name] = smoothed_matrix

    # 5. 可视化
    print("正在生成最终的相似度矩阵热图...")
    # 设置绘图风格和中文字体
    sns.set_style("white")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 为了公平比较，统一两个子图的颜色范围 (vmin, vmax)
    # 以平滑后的SCL模型的结果为基准，因为它通常有更集中的值域
    vmax = np.max(analysis_results['PM-SCL'])
    vmin = np.min(analysis_results['PM-SCL'])

    # 绘制基础模型的热图
    im1 = axes[0].imshow(analysis_results['PM'], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('(a) 基础模型 (PM)', fontsize=16)
    axes[0].set_xlabel('时空特征 (按PM2.5值排序)', fontsize=12)
    axes[0].set_ylabel('视觉特征 (按PM2.5值排序)', fontsize=12)

    # 绘制PM-SCL模型的热图
    title_scl = f'(b) PM-SCL模型 (高斯平滑, σ={args.smoothing_sigma})' if args.smoothing_sigma > 0 else '(b) PM-SCL模型'
    im2 = axes[1].imshow(analysis_results['PM-SCL'], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(title_scl, fontsize=16)
    axes[1].set_xlabel('时空特征 (按PM2.5值排序)', fontsize=12)
    axes[1].set_ylabel('') # 隐藏Y轴标签以保持图表简洁
    axes[1].set_yticks([])

    fig.suptitle('监督对比学习对跨模态特征相似度结构的影响', fontsize=20, y=0.98)
    
    # 为两个子图添加一个共享的颜色条
    cbar_label = '跨模态余弦相似度' + (' (平滑后)' if args.smoothing_sigma > 0 else '')
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存最终的图片
    save_path = f'final_similarity_matrix_smoothed_sigma{args.smoothing_sigma}.png'
    plt.savefig(save_path, dpi=300)
    print(f"最终优化后的可视化结果已保存至: {save_path}")
    
    plt.show()