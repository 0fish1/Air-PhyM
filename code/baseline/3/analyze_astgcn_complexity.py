import torch
import torch.nn as nn
import time
import numpy as np
from thop import profile

# Import the ASTGCN model and its config from your script
from astgcnbj import ASTGCNModel, BASELINE_CONFIG

# =============================================================================
#  Custom Handlers for thop
# =============================================================================
def zero_ops(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Custom handler for layers with zero FLOPs (e.g., ReLU)."""
    m.total_ops = torch.Tensor([0])
    m.total_params = torch.Tensor([0])

def count_norm(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Custom handler for normalization layers (e.g., LayerNorm)."""
    x = x[0]
    m.total_ops = torch.Tensor([2 * x.numel()]) 
    m.total_params = torch.Tensor([sum(p.numel() for p in m.parameters())])
# =============================================================================


def analyze_astgcn_cost():
    """
    Analyzes and prints the computational cost of the ASTGCN model.
    """
    # 1. Load Configuration and Model
    # -------------------------------------------------
    print("--- 1. Loading ASTGCN Model and Configuration ---")
    
    # Use the exact config from the astgcnbj.py script
    config = BASELINE_CONFIG

    device = torch.device(config["device"])
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU.")

    # Instantiate the ASTGCN model
    all_station_coords = torch.tensor(list(config["station_coords"].values()), dtype=torch.float32)
    model = ASTGCNModel(
        num_nodes=len(config["station_coords"]),
        in_features=6,
        num_timesteps=24,
        K=config["cheb_K"],
        num_chev_filters=config["num_chev_filters"],
        num_time_filters=config["num_time_filters"],
        station_coords=all_station_coords
    ).to(device)
    model.eval()
    print(f"Model 'ASTGCNModel' loaded successfully on '{device}'.")
    print("-" * 40)


    # 2. Create Dummy Inputs
    # -------------------------------------------------
    # ASTGCN takes the full pollution sequence [B, N, T, C]
    B_inf, B_train = 1, config["batch_size"]
    N = len(config["station_coords"]) # Total number of nodes (13)
    T = 24  # Number of timesteps
    F_poll = 6  # Number of pollution features (in_features)

    # Inference dummy data (Batch Size = 1)
    dummy_pollution_inf = torch.randn(B_inf, N, T, F_poll).to(device)
    inf_inputs = (dummy_pollution_inf,)

    # Training dummy data (Batch Size = 16)
    dummy_pollution_train = torch.randn(B_train, N, T, F_poll).to(device)
    dummy_target_train = torch.randn(B_train, 1).to(device)
    train_inputs = (dummy_pollution_train,)


    # 3. Calculate Trainable Parameters
    # -------------------------------------------------
    print("\n--- 2. Model Parameters ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f} M")
    print("-" * 40)


    # 4. Calculate FLOPs
    # -------------------------------------------------
    print("\n--- 3. FLOPs and MACs ---")
    
    custom_ops = {
        nn.ReLU: zero_ops,
        nn.LayerNorm: count_norm,
    }

    with torch.no_grad():
        macs, params = profile(
            model, 
            inputs=inf_inputs, 
            custom_ops=custom_ops,
            verbose=False
        )
    flops = macs * 2
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"MACs: {macs / 1e9:.2f} GMACs")
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
    
    step_times = []
    for stage in ["Warm-up", "Measurement"]:
        runs = 10 if stage == "Warm-up" else 50
        for _ in range(runs):
            start_time = time.time()
            optimizer.zero_grad()
            predictions = model(*train_inputs)
            loss = loss_fn(predictions, dummy_target_train)
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
    analyze_astgcn_cost()