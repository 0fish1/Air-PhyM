import torch
import torch.nn as nn
import time
import numpy as np
from thop import profile
from torch.nn.utils import remove_weight_norm

# Import the model, config, and the function we need to patch
import res_gcn 
from res_gcn import Res_GCN, RES_GCN_CONFIG

# =============================================================================
#  Custom Handlers for thop
# =============================================================================
def count_conv1d(m: nn.Conv1d, x: torch.Tensor, y: torch.Tensor):
    x = x[0]
    macs = m.in_channels * m.out_channels * m.kernel_size[0] * y.shape[2]
    if m.groups > 1: macs /= m.groups
    m.total_ops = torch.Tensor([macs])
    m.total_params = torch.Tensor([sum(p.numel() for p in m.parameters())])

def zero_ops(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    m.total_ops = torch.Tensor([0])
    m.total_params = torch.Tensor([0])

def count_norm(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    x = x[0]
    m.total_ops = torch.Tensor([2 * x.numel()]) 
    m.total_params = torch.Tensor([sum(p.numel() for p in m.parameters())])
# =============================================================================


def analyze_resgcn_cost():
    """
    Analyzes and prints the computational cost of the Res_GCN model.
    """
    # 1. Load Configuration and Model
    # -------------------------------------------------
    print("--- 1. Loading Res-GCN Model and Configuration ---")
    
    config = RES_GCN_CONFIG

    device = torch.device(config["device"])
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU.")

    model = Res_GCN(config).to(device)
    model.eval()
    print(f"Model 'Res_GCN' loaded successfully on '{device}'.")
    print("-" * 40)


    # 2. Create Dummy Inputs
    # -------------------------------------------------
    B_inf, B_train = 1, config["batch_size"]
    T = config["history_len"]
    N = config["num_stations"]
    F = config["input_features"]

    # Inference dummy data
    dummy_hist_inf = torch.randn(B_inf, T, N, F).to(device)
    dummy_img_inf = torch.randn(B_inf, 3, 224, 224).to(device)
    inf_inputs = (dummy_hist_inf, dummy_img_inf)

    # Training dummy data
    dummy_hist_train = torch.randn(B_train, T, N, F).to(device)
    dummy_img_train = torch.randn(B_train, 3, 224, 224).to(device)
    dummy_target_train = torch.randn(B_train, config["pred_len"]).to(device)
    train_inputs = (dummy_hist_train, dummy_img_train)


    # 3. Calculate Trainable Parameters
    # -------------------------------------------------
    print("\n--- 2. Model Parameters ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f} M")
    print("-" * 40)


    # 4. Calculate FLOPs
    # -------------------------------------------------
    print("\n--- 3. FLOPs and MACs ---")
    
    # Step 4.1: Remove weight_norm from TCN blocks
    print("Applying 'remove_weight_norm' to model for compatibility...")
    for module in model.modules():
        if hasattr(module, 'weight_g'):
            try: remove_weight_norm(module)
            except ValueError: pass
    print("Done.")

    # Step 4.2: Monkey patch the non-traceable fastdtw function
    print("Monkey patching 'calculate_dtw_adj' for FLOPs analysis...")
    original_dtw_func = res_gcn.calculate_dtw_adj
    def dummy_calculate_dtw_adj(batch_x, phi=1.0, epsilon=0.5):
        B, _, N, _ = batch_x.shape
        # Return a dummy adjacency matrix of the correct shape
        return torch.rand(B, N, N, device=batch_x.device)
    res_gcn.calculate_dtw_adj = dummy_calculate_dtw_adj
    print("Done.")

    custom_ops = {
        nn.Conv1d: count_conv1d,
        nn.ReLU: zero_ops,
        nn.LayerNorm: count_norm,
        nn.BatchNorm2d: count_norm,
    }

    with torch.no_grad():
        macs, params = profile(
            model, 
            inputs=inf_inputs, 
            custom_ops=custom_ops,
            verbose=False
        )
    
    # Restore the original function after profiling
    res_gcn.calculate_dtw_adj = original_dtw_func
    print("Restored original 'calculate_dtw_adj' function.")

    flops = macs * 2
    print(f"FLOPs (NN-only): {flops / 1e9:.2f} GFLOPs")
    print(f"MACs (NN-only): {macs / 1e9:.2f} GMACs")
    print("-" * 40)


    # 5. Measure Inference Latency
    # -------------------------------------------------
    print("\n--- 4. Inference Latency (Batch Size = 1) ---")
    print("Note: This measures REAL latency, including the slow, CPU-based fastdtw calculation.")
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        for _ in range(5): _ = model(*inf_inputs) # Short warm-up due to slow DTW
        
        with torch.no_grad():
            for i in range(20): # Reduced runs due to slow DTW
                starter.record()
                _ = model(*inf_inputs)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        
        avg_latency, std_latency = np.mean(timings), np.std(timings)
        print(f"Average Latency: {avg_latency:.2f} ms (+/- {std_latency:.2f} ms) per sample")
    else:
        print("Skipping latency test: Not on a CUDA device.")
    print("-" * 40)


    # 6. Measure Training Speed
    # -------------------------------------------------
    print(f"\n--- 5. Training Speed (Batch Size = {B_train}) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = torch.nn.MSELoss()
    
    step_times = []
    for stage in ["Warm-up", "Measurement"]:
        runs = 3 if stage == "Warm-up" else 10 # Reduced runs
        for _ in range(runs):
            start_time = time.time()
            optimizer.zero_grad()
            
            pred_seq = model(*train_inputs)
            pred_for_loss = pred_seq[:, 0].unsqueeze(1)
            # Create a dummy target with the correct shape for the loss function
            dummy_target_train_single_step = torch.randn(B_train, 1).to(device)
            loss = loss_fn(pred_for_loss, dummy_target_train_single_step)
            
            loss.backward()
            if device.type == 'cuda': torch.cuda.synchronize()
            optimizer.step()
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.time()
            if stage == "Measurement":
                step_times.append(end_time - start_time)

    avg_step_time, std_step_time = np.mean(step_times), np.std(step_times)
    print(f"Average Training Step Time: {avg_step_time:.4f} s (+/- {std_step_time:.4f} s)")
    print("-" * 40)


    # 7. Measure Peak GPU Memory
    # -------------------------------------------------
    print("\n--- 6. Peak GPU Memory Usage ---")
    model.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(*inf_inputs)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"Peak GPU Memory Allocated: {peak_memory:.2f} MB")
    else:
        print("Skipping GPU memory test: Not on a CUDA device.")
    print("-" * 40)


if __name__ == '__main__':
    analyze_resgcn_cost()