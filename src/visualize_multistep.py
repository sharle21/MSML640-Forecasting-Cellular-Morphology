import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, random_split
from src.train_trajectory_multistep import MultiStepTrajectoryDataset,MultiStepPredictor,DATA_FILE

# Configuration
Model_path = Path("trajectory_model_multistep.pth")
Hidden_Dimensions = 512 
Batch_size = 64
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#this returns a dataloader for the validation set
def get_validation_loader():
    
    dataset = MultiStepTrajectoryDataset(DATA_FILE)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    return DataLoader(val_dataset, batch_size=Batch_size, shuffle=False)

#this loads the trained model from file
def load_model():
    if not Model_path.exists():
        raise FileNotFoundError(
            f"Model file '{Model_path}' not found. Make sure training finished successfully.")

    model = MultiStepPredictor(hidden_dim=Hidden_Dimensions).to(Device)
    state_dict = torch.load(Model_path, map_location=Device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

#this collects real and predicted data from the validation set
def collect_real_and_pred(model, val_loader):
    real_24 = []
    real_72, real_2w, real_4w = [], [], []
    pred_72, pred_2w, pred_4w = [], [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(Device)
            outputs = model(inputs)

            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()

            real_24.append(inputs_np)
            real_72.append(targets_np[:, 0:384])
            real_2w.append(targets_np[:, 384:768])
            real_4w.append(targets_np[:, 768:])

            pred_72.append(outputs_np[:, 0:384])
            pred_2w.append(outputs_np[:, 384:768])
            pred_4w.append(outputs_np[:, 768:])

    
    R24 = np.vstack(real_24)
    R72 = np.vstack(real_72)
    R2W = np.vstack(real_2w)
    R4W = np.vstack(real_4w)

    P72 = np.vstack(pred_72)
    P2W = np.vstack(pred_2w)
    P4W = np.vstack(pred_4w)

    return R24, R72, R2W, R4W, P72, P2W, P4W

#this runs PCA on all real data and projects both real and predicted data into 2D
def run_pca(R24, R72, R2W, R4W, P72, P2W, P4W):
    all_real = np.vstack([R24, R72, R2W, R4W])
    pca = PCA(n_components=2)
    pca.fit(all_real)

    p24 = pca.transform(R24)
    r72 = pca.transform(R72)
    r2w = pca.transform(R2W)
    r4w = pca.transform(R4W)

    pred72 = pca.transform(P72)
    pred2w = pca.transform(P2W)
    pred4w = pca.transform(P4W)

    return p24, r72, r2w, r4w, pred72, pred2w, pred4w

#this plots the PCA results with trajectories
def plot_trajectories(p24, r72, r2w, r4w, pred72, pred2w, pred4w, num_to_plot=50):
    plt.figure(figsize=(12, 10))
    #Background scatter showing overall distribution
    plt.scatter(p24[:, 0], p24[:, 1], c="gray", alpha=0.1, s=10, label="24h (start)")
    plt.scatter(r4w[:, 0], r4w[:, 1], c="blue", alpha=0.2, s=10, label="4w real")
    plt.scatter(pred4w[:, 0], pred4w[:, 1], c="red", alpha=0.2, s=10, label="4w predicted")

    # Plot a subset of trajectories to keep things readable
    n = min(num_to_plot, p24.shape[0])
    for i in range(n):
        # Real path: 24h -> 72h -> 2w -> 4w
        x_real = [p24[i, 0], r72[i, 0], r2w[i, 0], r4w[i, 0]]
        y_real = [p24[i, 1], r72[i, 1], r2w[i, 1], r4w[i, 1]]
        plt.plot(x_real, y_real, "b-", alpha=0.3, linewidth=1)

        # Predicted path: 24h -> pred72 -> pred2w -> pred4w
        x_pred = [p24[i, 0], pred72[i, 0], pred2w[i, 0], pred4w[i, 0]]
        y_pred = [p24[i, 1], pred72[i, 1], pred2w[i, 1], pred4w[i, 1]]
        plt.plot(x_pred, y_pred, "r--", alpha=0.5, linewidth=1)

    real_line = mlines.Line2D([], [], color="blue", linestyle="-", label="Real trajectory")
    pred_line = mlines.Line2D([], [], color="red", linestyle="--", label="Predicted trajectory")

    plt.legend(handles=[real_line, pred_line], loc="best")
    plt.title(f"Cell Morphology Trajectories up to 4 Weeks (n={n})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("trajectory_multistep.png")
    print("Saved figure to 'trajectory_multistep.png'")


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file '{DATA_FILE}' not found.")

    val_loader = get_validation_loader()
    model = load_model()
    R24, R72, R2W, R4W, P72, P2W, P4W = collect_real_and_pred(model, val_loader)
    p24, r72, r2w, r4w, pred72, pred2w, pred4w = run_pca(R24, R72, R2W, R4W, P72, P2W, P4W)
    plot_trajectories(p24, r72, r2w, r4w, pred72, pred2w, pred4w)


if __name__ == "__main__":
    main()
