import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.train_trajectory_multistep import MultiStepTrajectoryDataset, MultiStepPredictor, DATA_FILE, HIDDEN_DIM

# --- CONFIG ---
# We use the quantized model we saved earlier
QUANTIZED_MODEL_PATH = "trajectory_model_quantized.pth"
BATCH_SIZE = 64

def check_accuracy():
    print("Verifying Quantized Model Accuracy...")
    
    # 1. Load Data
    full_dataset = MultiStepTrajectoryDataset(DATA_FILE)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Re-create Model Structure (FP32)
    model_fp32 = MultiStepPredictor(hidden_dim=HIDDEN_DIM).to('cpu')
    model_fp32.eval()
    
    # 3. Apply Quantization Structure
    # We must apply the "structure" of quantization before loading the quantized weights
    quantized_model = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    
    # 4. Load the INT8 Weights
    quantized_model.load_state_dict(torch.load(QUANTIZED_MODEL_PATH))
    
    # 5. Measure Accuracy
    cos = nn.CosineSimilarity(dim=1)
    total_sim = 0
    count = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            # Quantized models run on CPU
            inputs = inputs.to('cpu')
            targets = targets.to('cpu')
            
            outputs = quantized_model(inputs)
            
            # Measure Cosine Similarity
            sim = cos(outputs.view(-1, 384), targets.view(-1, 384)).mean().item()
            total_sim += sim * len(inputs)
            count += len(inputs)

    avg_acc = (total_sim / count) * 100
    
    print("\n" + "="*40)
    print(f"QUANTIZED MODEL ACCURACY")
    print("="*40)
    print(f"Accuracy: {avg_acc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    check_accuracy()
