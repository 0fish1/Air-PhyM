import torch
import torch.nn as nn
import time
import numpy as np
from thop import profile

# Import the STGCN model from your baseline script
from stgcn_baseline_bj import STGCN

# =============================================================================
#  Custom Handlers for thop (Simplified for STGCN)
# =============================================================================
def zero_ops(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Custom handler for layers with zero FLOPs (e.g., ReLU)."""
    m.total_ops = torch.Tensor([0])
    m.total_params = torch.Tensor([0])
# =============================================================================


def analyze_stgcn_cost():
    """
    Analyzes and prints the computational cost of the STGCN model.
    """
    # 1. Load Configuration and Model
    # -------------------------------------------------
    print("--- 1. Loading STGCN Model and Configuration ---")
    
    # Use a simple config matching the stgcn_baseline_bj.py script
    config = {
        "site_nums": 12,
        "history_hours": 24,
        "batch_size": 16, # Define for training speed test
        "learning_rate": 1e-3, # Define for optimizer
        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
    }

    device = torch.device(config["device"])
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU.")

    # Instantiate the STGCN model
    model = STGCN(c_in=6, c_out=1, k_top=3).to(device)
    model.eval()
    print(f"Model 'STGCN' loaded successfully on '{device}'.")
    print("-" * 40)


    # 2. Create Dummy Inputs
    # -------------------------------------------------
    # STGCN only takes the pollution sequence as input
    B_inf, B_train = 1, config["batch_size"]
    T = config["history_hours"]
    N = config["site_nums"]
    F_poll = 6  # Number of pollution features (c_in)

    # Inference dummy data (Batch Size = 1)
    dummy_pollution_inf = torch.randn(B_inf, N, T, F_poll).to(device)
    # The input must be a tuple for the profile function
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
    
    # Define custom handlers needed for this model
    custom_ops = {
        nn.ReLU: zero_ops,
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
    # Warm-up and Measurement loops
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
    analyze_stgcn_cost()