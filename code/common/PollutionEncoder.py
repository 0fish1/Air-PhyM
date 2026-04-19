from pathlib import Path; import sys
COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path: sys.path.insert(0, str(COMMON_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


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
        self.num_channels = num_channels
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


class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, alpha=-0.2, phys_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.phys_bias = phys_bias

        self.W = nn.Linear(in_dim, out_dim)

        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim, 1))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.self_bias = nn.Parameter(torch.zeros(num_heads))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj_phys, save_dir=None, batch_idx=0):
        """
        x: [B, N+1, in_dim] - node embeddings (N neighbors + 1 target)
        adj_phys: [B, N] - physical weights from neighbors to target
        Returns: [B, N+1, out_dim] - updated embeddings for ALL nodes
        """
        B, N_total, D = x.shape
        N = N_total - 1

        h = self.W(x)  # [B, N+1, out_dim]
        h_heads = h.view(B, N_total, self.num_heads, self.head_dim)

        if self.phys_bias:
            adj_phys_softmax = F.softmax(adj_phys, dim=-1)  # [B, N]
            phys_bias_row = torch.zeros(B, N_total, device=x.device)
            phys_bias_row[:, 1:] = adj_phys_softmax

        attn_heads = []
        for head in range(self.num_heads):
            Wh_i = h_heads[:, :, head, :].unsqueeze(2)  # [B, N+1, 1, d]
            Wh_j = h_heads[:, :, head, :].unsqueeze(1)  # [B, 1, N+1, d]
            Wh_i_exp = Wh_i.expand(B, N_total, N_total, self.head_dim)
            Wh_j_exp = Wh_j.expand(B, N_total, N_total, self.head_dim)
            concat = torch.cat([Wh_i_exp, Wh_j_exp], dim=-1)

            a_head = self.a[head]
            e_flat = torch.matmul(concat.view(B * N_total * N_total, -1), a_head)
            e = e_flat.squeeze(-1).view(B, N_total, N_total)
            e = self.leaky_relu(e)

            if self.phys_bias:
                bias_row = torch.zeros(B, N_total, N_total, device=x.device)
                bias_row[:, 0, 1:] = adj_phys_softmax
                bias_row[:, 0, 0] = self.self_bias[head]
                e = e + self.gamma * bias_row

            attn = F.softmax(e, dim=-1)
            attn = self.dropout(attn)
            attn_heads.append(attn)

        attn_heads = torch.stack(attn_heads, dim=-1)

        out_heads = []
        for head in range(self.num_heads):
            attn_head = attn_heads[:, :, :, head]
            h_head = h_heads[:, :, head, :]
            out_head = torch.matmul(attn_head, h_head)
            out_heads.append(out_head)

        out = torch.cat(out_heads, dim=-1)  # [B, N+1, out_dim]

        return self.norm(out)


class PollutionGATEncoder(nn.Module):
    def __init__(self, in_pollution=6, hidden_dim=128, tcn_channels=[8,16,32,64], weather_dim=5, gat_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        embed_dim = tcn_channels[-1] + hidden_dim

        self.tcn_poll = TemporalConvNet(num_inputs=in_pollution, num_channels=tcn_channels)
        self.lstm_weather = nn.LSTM(input_size=weather_dim, hidden_size=hidden_dim, batch_first=True)
        self.proj_embed = nn.Linear(embed_dim, hidden_dim)

        # learnable embedding for target station (replaces zero vector)
        tcn_out_dim = tcn_channels[-1]
        self.target_poll_embed = nn.Parameter(torch.randn(1, tcn_out_dim))

        # 2-layer GAT with residual connection
        self.gat1 = SimpleGATLayer(in_dim=hidden_dim, out_dim=hidden_dim, num_heads=gat_heads, phys_bias=True)
        self.gat2 = SimpleGATLayer(in_dim=hidden_dim, out_dim=hidden_dim, num_heads=gat_heads, phys_bias=True)

    def forward(self, pollution, weather, coords, A_phys, save_dir=None, batch_idx=0):
        B, N, T, F_poll = pollution.shape  # [B, N_neighbors, T, 6]
        _, T_w, F_w = weather.shape  # [B, T, 5]

        # 1. pollution embedding
        pollution = pollution.permute(0, 1, 3, 2).reshape(B * N, F_poll, T)
        poll_out = self.tcn_poll(pollution)[:, :, -1]  # [B*N, tcn_out_dim]
        poll_embed = poll_out.view(B, N, -1)  # [B, N_neighbors, tcn_out_dim]

        # 2. weather embedding
        _, (h_weather, _) = self.lstm_weather(weather)
        weather_embed = h_weather[-1]  # [B, hidden_dim]
        weather_embed = weather_embed.unsqueeze(1).expand(B, N + 1, -1)  # [B, N+1, hidden_dim]

        # 3. target station: learnable embedding (replaces zero vector)
        target_poll = self.target_poll_embed.expand(B, -1).unsqueeze(1)  # [B, 1, tcn_out_dim]
        full_poll_embed = torch.cat([target_poll, poll_embed], dim=1)  # [B, N+1, tcn_out_dim]

        # 4. concat embeddings
        node_embed = torch.cat([full_poll_embed, weather_embed], dim=-1)  # [B, N+1, embed_dim]

        # 5. project
        node_embed = self.proj_embed(node_embed)  # [B, N+1, hidden_dim]
        node_embed_before = node_embed  # save for residual

        # 6. 2-layer GAT aggregation with residual
        node_embed = self.gat1(node_embed, A_phys)  # [B, N+1, hidden_dim] — neighbor info flows
        node_embed = node_embed + node_embed_before    # residual connection
        node_embed = self.gat2(node_embed, A_phys)  # [B, N+1, hidden_dim] — final aggregation
        target_feat = node_embed[:, 0, :]  # [B, hidden_dim] — target station output

        return target_feat