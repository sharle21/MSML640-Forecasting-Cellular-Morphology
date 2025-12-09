import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, random_split
from src.train_trajectory import CellTrajectoryDataset, TrajectoryPredictor, DATA_FILE

Batch_size = 64
Hidden_Dimensions = 1024  
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#this returns a dataloader for the validation set
def get_validation_loader(): 
    dataset = CellTrajectoryDataset(DATA_FILE)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    return DataLoader(val_dataset, batch_size=Batch_size, shuffle=False)

#this loads the trained model from file
def load_model():
    model = TrajectoryPredictor(hidden_dim=Hidden_Dimensions).to(Device)
    state_dict = torch.load("trajectory_model.pth", map_location=Device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

#this collects the actual and predicted 72h vectors from the validation set
def collect_predictions(model, val_loader):
    actual_list = []
    pred_list = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(Device)
            outputs = model(inputs)

            actual_list.append(targets.cpu().numpy())
            pred_list.append(outputs.cpu().numpy())

    actual_72h = np.vstack(actual_list)
    predicted_72h = np.vstack(pred_list)

    return actual_72h, predicted_72h

# PCA on the actual data and projects both actual and predicted data into 2D
def run_pca(actual_72h, predicted_72h):
    pca = PCA(n_components=2)
    pca.fit(actual_72h)
    actual_2d = pca.transform(actual_72h)
    pred_2d = pca.transform(predicted_72h)

    return actual_2d, pred_2d

#tplot the PCA results
def plot_pca(actual_2d, pred_2d, max_pairs=100):
    plt.figure(figsize=(10, 8))
    plt.scatter(actual_2d[:, 0], actual_2d[:, 1],c="blue", alpha=0.5,s=10, label="Actual 72h")
    plt.scatter(pred_2d[:, 0], pred_2d[:, 1],c="red",alpha=0.5,s=10,label="Predicted 72h")

    #Draw lines between corresponding points for a subset, to avoid clutter
    n = min(max_pairs, actual_2d.shape[0])
    for i in range(n):
        x_vals = [actual_2d[i, 0], pred_2d[i, 0]]
        y_vals = [actual_2d[i, 1], pred_2d[i, 1]]
        plt.plot(x_vals, y_vals, "k-", alpha=0.2, linewidth=0.5)

    plt.legend()
    plt.title("Trajectory Prediction (PCA projection)\nBlue = actual, Red = predicted")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("prediction_vis.png")
    print("Saved figure to 'prediction_vis.png'")


def main():
    val_loader = get_validation_loader()
    model = load_model()
    actual_72h, predicted_72h = collect_predictions(model, val_loader)
    actual_2d, pred_2d = run_pca(actual_72h, predicted_72h)
    plot_pca(actual_2d, pred_2d)


if __name__ == "__main__":
    main()
