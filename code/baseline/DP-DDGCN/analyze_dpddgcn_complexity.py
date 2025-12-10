import torch
import torch.nn as nn
import time
import numpy as np
from thop import profile

# Import the DP_DDGCN model and its config from your script
from dp_ddgcn import DP_DDGCN, DP_DDGCN_CONFIG

# =============================================================================
#  Custom Handlers for thop
# =============================================================================
def zero_ops(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Custom handler for layers with zero FLOPs (e.g., ReLU)."""
    m.total_ops = torch.Tensor([0])
    m.total_params = torch.Tensor([0])
# =============================================================================


def analyze_dpddgcn_cost():
    """
    Analyzes and prints the computational cost of the DP_DDGCN model.
    """
    # 1. Load Configuration and Model
    # -------------------------------------------------
    print("--- 1. Loading DP-DDGCN Model and Configuration ---")
    
    config = DP_DDGCN_CONFIG

    device = torch.device(config["device"])
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU.")

    model = DP_DDGCN(config).to(device)
    model.eval()
    print(f"Model 'DP_DDGCN' loaded successfully on '{device}'.")
    print("-" * 40)


    # 2. Create Dummy Inputs
    # -------------------------------------------------
    B_inf, B_train = 1, config["batch_size"]
    T_h = config["history_len"]
    N = config["num_stations"]
    F_p = config["pollutant_features"]
    F_w = 5 # Assuming 5 weather features based on other scripts

    # Inference dummy data (Batch Size = 1)
    dummy_poll_inf = torch.randn(B_inf, T_h, N, F_p).to(device)
    dummy_wea_inf = torch.randn(B_inf, T_h, F_w).to(device)
    inf_inputs = (dummy_poll_inf, dummy_wea_inf)

    # Training dummy data
    dummy_poll_train = torch.randn(B_train, T_h, N, F_p).to(device)
    dummy_wea_train = torch.randn(B_train, T_h, F_w).to(device)
    # The model predicts for all stations, but loss is on one.
    # We'll mimic this by creating a target for a single station's prediction.
    dummy_target_train = torch.randn(B_train, 1).to(device)
    train_inputs = (dummy_poll_train, dummy_wea_train)


    # 3. Calculate Trainable Parameters
    # -------------------------------------------------
    print("\n--- 2. Model Parameters ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f} M")
    print("-" * 40)


    # 4. Calculate FLOPs
    # -------------------------------------------------
    print("\n--- 3. FLOPs and MACs ---")
    
    # !!! IMPORTANT WARNING ABOUT FOR LOOPS !!!
    print("\nWARNING: The DP-DDGCN model contains a for-loop in its forward pass.")
    print("         'thop' cannot trace through loops and will likely only profile")
    print("         a single iteration. The reported FLOPs/MACs will be an")
    print("         underestimation of the true computational cost.\n")

    custom_ops = { nn.ReLU: zero_ops }

    with torch.no_grad():
        macs, params = profile(
            model, 
            inputs=inf_inputs, 
            custom_ops=custom_ops,
            verbose=False
        )
    flops = macs * 2
    print(f"FLOPs (Underestimated): {flops / 1e9:.2f} GFLOPs")
    print(f"MACs (Underestimated): {macs / 1e9:.2f} GMACs")
    print("-" * 40)


    # 5. Measure Inference Latency
    # -------------------------------------------------
    print("\n--- 4. Inference Latency (Batch Size = 1) ---")
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        for _ in range(20): _ = model(*inf_inputs) # Warm-up
        
        with torch.no_grad():
            for _ in range(100):
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
    
    # Index of the station used for loss calculation in the original script
    NEIGHBOR_FOR_TARGET_IDX = 7 
    
    step_times = []
    for stage in ["Warm-up", "Measurement"]:
        runs = 10 if stage == "Warm-up" else 50
        for _ in range(runs):
            start_time = time.time()
            optimizer.zero_grad()
            
            pred_seq = model(*train_inputs)
            # Mimic the loss calculation from the training script
            pred_for_loss = pred_seq[:, 0, NEIGHBOR_FOR_TARGET_IDX, 0].unsqueeze(1)
            loss = loss_fn(pred_for_loss, dummy_target_train)
            
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
    analyze_dpddgcn_cost()