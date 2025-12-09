import torch
import torch.nn as nn
import os
import time
import numpy as np
from src.train_trajectory_multistep import MultiStepPredictor, HIDDEN_DIM

MODEL_PATH = "trajectory_model_multistep.pth"
QUANTIZED_PATH = "trajectory_model_quantized.pth"

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size_mb

def run_optimization():
    

    model = MultiStepPredictor(hidden_dim=HIDDEN_DIM).to('cpu')
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    input_tensor = torch.randn(1, 384) # One single cell
    
    start = time.time()
    for _ in range(1000):
        _ = model(input_tensor)
    end = time.time()
    avg_time_fp32 = (end - start) / 1000 * 1000 # ms
    
    size_fp32 = print_size_of_model(model)
    print(f"\nOriginal (FP32): {size_fp32:.2f} MB | Latency: {avg_time_fp32:.4f} ms")

    # converting to int8
    print("\n Applying Dynamic INT8 Quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear},  # Quantize the Linear layers
        dtype=torch.qint8
    )
    

    start = time.time()
    for _ in range(1000):
        _ = quantized_model(input_tensor)
    end = time.time()
    avg_time_int8 = (end - start) / 1000 * 1000 # ms
    
    size_int8 = print_size_of_model(quantized_model)
    print(f"Optimized (INT8): {size_int8:.2f} MB | Latency: {avg_time_int8:.4f} ms")
    
  #Summary Stats
    print("-" * 30)
    print(f"Size Reduction: {size_fp32 / size_int8:.1f}x smaller")
    print(f"Speedup:        {avg_time_fp32 / avg_time_int8:.1f}x faster")
    print("-" * 30)
    

    torch.save(quantized_model.state_dict(), QUANTIZED_PATH)


if __name__ == "__main__":
    run_optimization()
