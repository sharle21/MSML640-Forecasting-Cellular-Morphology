import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from src.train_trajectory_multistep import MultiStepTrajectoryDataset, MultiStepPredictor, DATA_FILE, DEVICE

# --- CONFIG ---
MODEL_PATH = "trajectory_model_multistep.pth"
HIDDEN_DIM = 512
BATCH_SIZE = 64

def test_robustness():
    print("Starting Robustness Analysis (Bonus Task 1)...")
    
    full_dataset = MultiStepTrajectoryDataset(DATA_FILE)
    train_size = int(0.8 * len(full_dataset))
    _, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiStepPredictor(hidden_dim=HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    cos = torch.nn.CosineSimilarity(dim=1)
    
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    accuracies = []
    
    print("\n--- Stress Test Results ---")
    for noise in noise_levels:
        total_sim = 0
        count = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                
                # Add Noise (Simulating bad microscope focus)
                noise_tensor = torch.randn_like(inputs) * noise
                noisy_inputs = inputs + noise_tensor
                
                # Predict
                outputs = model(noisy_inputs)
                
                # Compare 4W prediction (last 384 dims) to Real 4W (last 384 dims)
                pred_4w = outputs[:, 768:]
                real_4w = targets[:, 768:]
                
                sim = cos(pred_4w, real_4w).mean().item()
                total_sim += sim * len(inputs)
                count += len(inputs)
                
        avg_acc = (total_sim / count) * 100
        accuracies.append(avg_acc)
        print(f"Noise Level: {noise:.2f} | Accuracy: {avg_acc:.2f}%")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, accuracies, marker='o', linestyle='-', color='red')
    plt.title("Model Robustness to Input Noise")
    plt.xlabel("Noise Level (Std Dev)")
    plt.ylabel("Prediction Accuracy (%)")
    plt.grid(True)
    plt.savefig("robustness_analysis.png")
    print("✅ Saved robustness_analysis.png")

if __name__ == "__main__":
    test_robustness()
