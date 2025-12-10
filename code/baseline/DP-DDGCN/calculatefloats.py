def calculate_dpddgcn_flops():
    """
    手动计算DP-DDGCN模型的FLOPs。
    """
    # --- 配置 ---
    B = 1
    N = 9
    F_p = 6
    H = 64
    T_h = 6
    T_p = 1

    # --- MACs计算函数 ---
    def macs_linear(b, n_in, n_out):
        return b * n_in * n_out

    def macs_graph_conv(b, n_nodes, n_in, n_out):
        # bmm(adj, x) + linear(x)
        return b * (n_nodes * n_nodes * n_in + n_nodes * n_in * n_out)

    def macs_gru_step(b, n_nodes, h_dim):
        # 3 * (W_i*x + W_h*h) -> 6 * H*H per node
        return b * n_nodes * (6 * h_dim * h_dim)

    # --- Encoder ---
    macs_gcn_out_enc = macs_graph_conv(B, N, F_p, H)
    macs_gcn_in_enc = macs_graph_conv(B, N, H, H)
    macs_dp_ddgcb_enc = macs_gcn_out_enc + macs_gcn_in_enc
    macs_gru_enc = macs_gru_step(B, N, H)
    macs_per_encoder_step = macs_dp_ddgcb_enc + macs_gru_enc
    total_macs_encoder = T_h * macs_per_encoder_step

    # --- Decoder ---
    macs_mlp = macs_linear(B * N, H, 32) + macs_linear(B * N, 32, F_p)
    macs_gcn_out_dec = macs_graph_conv(B, N, F_p, H)
    macs_gcn_in_dec = macs_graph_conv(B, N, H, H)
    macs_dp_ddgcb_dec = macs_gcn_out_dec + macs_gcn_in_dec
    macs_gru_dec = macs_gru_step(B, N, H)
    macs_per_decoder_step = macs_mlp + macs_dp_ddgcb_dec + macs_gru_dec + macs_mlp
    total_macs_decoder = T_p * macs_per_decoder_step
    
    # --- 总计 ---
    total_macs = total_macs_encoder + total_macs_decoder
    total_flops = 2 * total_macs

    print("--- DP-DDGCN 手动计算结果 ---")
    print(f"Encoder MACs: {total_macs_encoder:,.0f}")
    print(f"Decoder MACs: {total_macs_decoder:,.0f}")
    print("-" * 30)
    print(f"总 MACs: {total_macs:,.0f}")
    print(f"总 FLOPs: {total_flops:,.0f}")
    print(f"总 GFLOPs: {total_flops / 1e9:.6f}")

if __name__ == '__main__':
    calculate_dpddgcn_flops()