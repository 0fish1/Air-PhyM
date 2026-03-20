# analyze_contribution.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from torch.utils.data import DataLoader, Subset

# 导入您项目中的模块
from model import AirQualityModel
from dataset import AirQualityDataset
from configs import experiment_configs

def analyze(args):
    """主分析函数"""
    # --- 1. 加载配置和模型 ---
    config = experiment_configs.get(args.exp)
    if not config:
        print(f"Error: Experiment '{args.exp}' not found.")
        return
    
    if config.get("fusion_type") != "symmetrical_gated_attention":
        print(f"Warning: This analysis is designed for 'symmetrical_gated_attention' fusion.")
        print(f"The selected experiment '{args.exp}' uses '{config.get('fusion_type')}' fusion.")

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = AirQualityModel(config).to(device)
    
    # <<< --- START OF MODIFICATION --- >>>
    # 核心修复：像 run_experiment.py 一样构建正确的模型路径
    # 我们不再使用 config 中默认的 "./checkpoints"，而是根据实验名称动态生成路径
    save_dir = f"experiments/{args.exp}"
    model_path = os.path.join(save_dir, "best_model.pth")
    # <<< --- END OF MODIFICATION --- >>>

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        print("Please ensure you have successfully trained the model for the experiment:")
        print(f"  python run_experiment.py --exp {args.exp}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # --- 2. 加载数据集并找到测试集索引 ---
    # 为了复现train.py中的数据集划分，我们使用相同的种子
    full_dataset = AirQualityDataset(config["pkl_file"], config["distances"], config)
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # 使用固定的生成器来确保每次划分都一样
    generator = torch.Generator().manual_seed(config["seed"])
    _, _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"Loaded test set with {len(test_dataset)} samples.")

    # --- 3. 筛选代表性样本 ---
    print("Finding representative samples from the test set...")
    targets_with_indices = []
    # test_dataset.indices 包含了它在 full_dataset 中的原始索引
    for i in range(len(test_dataset)):
        # test_dataset[i] 会返回 full_dataset 中对应索引的数据
        _, _, _, _, target = test_dataset[i]
        original_index = test_dataset.indices[i]
        targets_with_indices.append((target.item(), original_index))

    sorted_samples = sorted(targets_with_indices, key=lambda x: x[0])
    
    # 选取污染最轻和最重的 N 个样本
    num_samples_to_show = 3
    clean_samples = sorted_samples[:num_samples_to_show]
    polluted_samples = sorted_samples[-num_samples_to_show:]

    # --- 4. 执行推理并获取贡献度 ---
    def run_inference_and_get_alpha(original_index):
        """对单个样本进行推理并返回结果"""
        imgs, pollution, weather, adj, target = full_dataset[original_index]
        
        # 增加 batch 维度并移动到设备
        imgs = imgs.unsqueeze(0).to(device)
        pollution = pollution.unsqueeze(0).to(device)
        weather = weather.unsqueeze(0).to(device)
        adj_hybrid, adj_phys = adj
        adj_mask = (adj_hybrid.unsqueeze(0).to(device), adj_phys.unsqueeze(0).to(device))
        
        with torch.no_grad():
            prediction, alpha = model(imgs, pollution, weather, adj_mask, return_contribution=True)
        
        return {
            "index": original_index,
            "ground_truth": target.item(),
            "prediction": prediction.item(),
            "image_weight": alpha.item(),
            "numeric_weight": 1.0 - alpha.item()
        }

    clean_results = [run_inference_and_get_alpha(idx) for _, idx in clean_samples]
    polluted_results = [run_inference_and_get_alpha(idx) for _, idx in polluted_samples]

    # --- 5. 可视化结果 ---
    def visualize_results(results, title):
        """创建条形图来可视化贡献度"""
        num_results = len(results)
        fig, axes = plt.subplots(num_results, 1, figsize=(8, 3 * num_results), sharex=True)
        if num_results == 1: axes = [axes] # 保证axes是可迭代的
        
        fig.suptitle(title, fontsize=16, y=1.02)
        
        for i, res in enumerate(results):
            ax = axes[i]
            labels = ['Image Modality', 'Spatio-Temporal Modality']
            weights = [res['image_weight'], res['numeric_weight']]
            
            bars = ax.bar(labels, weights, color=['skyblue', 'salmon'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Contribution Weight')
            ax.set_title(f"Sample Index: {res['index']} | Ground Truth: {res['ground_truth']:.2f} | Prediction: {res['prediction']:.2f}")

            # 在条形图上显示数值
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # 保存图像而不是直接显示
        output_filename = f"{save_dir}/{title.replace(' ', '_').lower()}.png"
        plt.savefig(output_filename)
        print(f"Analysis plot saved to: {output_filename}")
        plt.close(fig) # 关闭图像，防止在无GUI环境下报错

    print("\n--- Analysis Results ---")
    visualize_results(clean_results, "Contribution Analysis on Clean Days")
    visualize_results(polluted_results, "Contribution Analysis on Heavy Pollution Days")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dynamic Multimodal Contribution Analysis")
    parser.add_argument('--exp', type=str, required=True,
                        choices=list(experiment_configs.keys()),
                        help="The experiment name to analyze (e.g., PM or PM_SCL).")
    args = parser.parse_args()
    
    analyze(args)