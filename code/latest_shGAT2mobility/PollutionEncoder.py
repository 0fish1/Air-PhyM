import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import uuid
from torch.nn.utils import weight_norm

# 原TCN组件保留（用于污染物嵌入）
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.3):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.num_channels = num_channels  # 新增：存储通道列表
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 新增：位置嵌入MLP
class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),  # 输入经纬度[lat, lon]
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

    def forward(self, coords):
        # coords: [B, N, 2] 或 [B, 2]
        return self.mlp(coords.float())

# 新增：简单GAT层（单层多头，纯PyTorch）
class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, alpha=-0.2, phys_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.phys_bias = phys_bias

        self.W = nn.Linear(in_dim, out_dim)
        
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim, 1))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj_phys, save_dir=None, batch_idx=0):
        """
        x: [B, N=13, in_dim] - 节点嵌入
        adj_phys: [B, 12] - 周边到目标的物理权重
        save_dir: str - 保存热图的目录（可选）
        batch_idx: int - 当前批次索引，用于文件名
        """
        B, N, D = x.shape
        assert N == 10

        # Project
        h = self.W(x)  # [B, N, out_dim]
        h_heads = h.view(B, N, self.num_heads, self.head_dim)  # [B, N, heads, head_dim]

        # Build adj_matrix [B, N, N]
        device = x.device
        adj_matrix = torch.ones(B, N, N, device=device) * 0.1
        adj_matrix[:, 0, 0] = 1.0
        for i in range(1, N):
            adj_matrix[:, 0, i] = adj_phys[:, i-1]
            adj_matrix[:, i, 0] = adj_phys[:, i-1]
        for i in range(1, N):
            for j in range(1, N):
                if i != j:
                    adj_matrix[:, i, j] = adj_phys[:, i-1] * adj_phys[:, j-1] * 0.5

        # 可视化：仅对第1个批次样本保存热图
        # if save_dir and batch_idx == 0:
        #     os.makedirs(save_dir, exist_ok=True)
            
        #     # 可视化 adj_phys [B, 12]
        #     plt.figure(figsize=(6, 2))
        #     plt.imshow(adj_phys[0:1].detach().cpu().numpy(), cmap='viridis', aspect='auto')
        #     plt.colorbar(label='A_phys Weight')
        #     plt.title('Physical Weights (A_phys) for Batch 0')
        #     plt.xlabel('Station Index (0-11)')
        #     plt.ylabel('Batch')
        #     plt.savefig(os.path.join(save_dir, f'adj_phys_batch_{batch_idx}.png'))
        #     plt.close()

        #     # 可视化 adj_matrix [13, 13]（取batch 0）
        #     plt.figure(figsize=(8, 8))
        #     plt.imshow(adj_matrix[0].detach().cpu().numpy(), cmap='viridis')
        #     plt.colorbar(label='A_matrix Weight')
        #     plt.title('Adjacency Matrix for Batch 0')
        #     plt.xlabel('Node Index (0=target, 1-12=neighbors)')
        #     plt.ylabel('Node Index (0=target, 1-12=neighbors)')
        #     plt.savefig(os.path.join(save_dir, f'adj_matrix_batch_{batch_idx}.png'))
        #     plt.close()

        # Attention per head
        attn_heads = []
        for head in range(self.num_heads):
            Wh_i = h_heads[:, :, head, :].unsqueeze(2)  # [B, N, 1, d]
            Wh_j = h_heads[:, :, head, :].unsqueeze(1)  # [B, 1, N, d]
            Wh_i_exp = Wh_i.expand(B, N, N, self.head_dim)  # [B, N, N, d]
            Wh_j_exp = Wh_j.expand(B, N, N, self.head_dim)  # [B, N, N, d]
            concat = torch.cat([Wh_i_exp, Wh_j_exp], dim=-1)  # [B, N, N, 2*d]

            a_head = self.a[head]  # [2*d, 1]
            e_flat = torch.matmul(concat.view(B * N * N, -1), a_head)  # [B*N*N, 1]
            e = e_flat.squeeze(-1).view(B, N, N)  # [B, N, N]
            e = self.leaky_relu(e)

            if self.phys_bias:
                eps = 1e-6
                e = e + torch.log(adj_matrix + eps)

            attn = F.softmax(e, dim=-1)
            attn = self.dropout(attn)
            attn_heads.append(attn)

        # Stack [B, N, N, heads]
        attn_heads = torch.stack(attn_heads, dim=-1)  # [B, N, N, heads]

        # Compute out per head
        out_heads = []
        for head in range(self.num_heads):
            attn_head = attn_heads[:, :, :, head]  # [B, N, N]
            h_head = h_heads[:, :, head, :]  # [B, N, d]
            out_head = torch.matmul(attn_head, h_head)  # [B, N, d]
            out_heads.append(out_head)

        # Cat over heads
        out = torch.cat(out_heads, dim=-1)  # [B, N, out_dim]

        target_out = out[:, 0, :]  # [B, out_dim]
        return self.norm(target_out)

# 新编码器：TCN (污染物) + LSTM (气象) + Pos (位置) + GAT
class PollutionGATEncoder(nn.Module):
    def __init__(self, in_pollution=6, hidden_dim=128, tcn_channels=[8,16,32,64], weather_dim=5, pos_dim=8, gat_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        embed_dim = tcn_channels[-1] + hidden_dim + pos_dim  # 64 (poll) + 128 (weather) + 8 (pos) = 200

        self.tcn_poll = TemporalConvNet(num_inputs=in_pollution, num_channels=tcn_channels)
        self.lstm_weather = nn.LSTM(input_size=weather_dim, hidden_size=hidden_dim, batch_first=True)
        self.pos_embed = PositionalEmbedding(pos_dim)
        self.proj_embed = nn.Linear(embed_dim, hidden_dim)
        self.gat = SimpleGATLayer(in_dim=hidden_dim, out_dim=hidden_dim, num_heads=gat_heads, phys_bias=True)

    def forward(self, pollution, weather, coords, A_phys, save_dir=None, batch_idx=0):
        B, N, T, F_poll = pollution.shape  # [B, 12, T, 6]
        _, T_w, F_w = weather.shape  # [B, T, 5]

        # 1. 污染物嵌入
        pollution = pollution.permute(0, 1, 3, 2).reshape(B * N, F_poll, T)
        poll_out = self.tcn_poll(pollution)[:, :, -1]  # [B*12, 64]
        poll_embed = poll_out.view(B, N, -1)  # [B, 12, 64]

        # 2. 气象嵌入
        _, (h_weather, _) = self.lstm_weather(weather)
        weather_embed = h_weather[-1]  # [B, 128]
        weather_embed = weather_embed.unsqueeze(1).expand(B, N+1, -1)  # [B, 13, 128]

        # 3. 位置嵌入
        pos_embed = self.pos_embed(coords)  # [B, 13, 8]

        # 4. 目标站点：零填充污染物
        tcn_out_dim = self.tcn_poll.num_channels[-1]  # 64
        target_poll = torch.zeros(B, 1, tcn_out_dim, device=pollution.device)
        full_poll_embed = torch.cat([target_poll, poll_embed], dim=1)  # [B, 13, 64]

        # 5. 拼接嵌入
        node_embed = torch.cat([full_poll_embed, weather_embed, pos_embed], dim=-1)  # [B, 13, 200]

        # 6. 投影
        node_embed = self.proj_embed(node_embed)  # [B, 13, 128]

        # 7. GAT聚合
        target_feat = self.gat(node_embed, A_phys, save_dir=save_dir, batch_idx=batch_idx)  # [B, 128]

        return target_feat