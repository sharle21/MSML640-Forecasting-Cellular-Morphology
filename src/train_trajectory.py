import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Configuration
DATA_FILE = "openphenom_embeddings_full.pt"
Model_save_path = "trajectory_model.pth"
Batch_size = 64
Epochs = 50
learning_rate = 0.001
Hidden_dim = 1024
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Starting training on {Device}")

#Daraset class for 24h -> 72h trajectory prediction
class CellTrajectoryDataset(Dataset):
   
    def __init__(self, pt_file):
        self.data = torch.load(pt_file)

        self.valid_data = []
        for item in self.data:
            e24 = item["embedding_24h"]
            e72 = item["embedding_72h"]

            if torch.is_tensor(e24) and torch.is_tensor(e72):
                if e24.shape[0] == 384 and e72.shape[0] == 384:
                    self.valid_data.append(item)

        discarded = len(self.data) - len(self.valid_data)
        print(f"Loaded {len(self.valid_data)} valid pairs (discarded {discarded})")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        x = item["embedding_24h"].float()
        y = item["embedding_72h"].float()
        return x, y


class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=1024):
        super(TrajectoryPredictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)

#Cosine similarity loss function
class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, pred, target):
        sim = self.cosine(pred, target)
        loss = 1 - sim.mean()
        return loss


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run the extraction step first.")
        exit()

    #Prepare data for training and validation
    dataset = CellTrajectoryDataset(DATA_FILE)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=Batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Batch_size, shuffle=False)

    # Model and optimization setup
    model = TrajectoryPredictor(hidden_dim=Hidden_dim).to(Device)
    criterion = CosineLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print(f"\nTraining for {Epochs} epochs with hidden_dim = {Hidden_dim}...\n")

    #Training
    for epoch in range(Epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(Device), targets.to(Device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(Device), targets.to(Device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), Model_save_path)
            saved_msg = " (model saved)"
        else:
            saved_msg = ""

        print(
            f"Epoch {epoch + 1}/{Epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"LR: {current_lr:.6f}{saved_msg}"
        )

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")

    final_sim = 1 - best_val_loss
    print(f"Final cosine similarity (approx. accuracy): {final_sim * 100:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine loss (lower is better)")
    plt.title(f"Training curve (final similarity: {final_sim:.2f})")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curve.png")
    print("Saved training curve to 'training_curve.png'")
