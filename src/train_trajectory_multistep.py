import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

# Configuration
DATA_FILE = "openphenom_embeddings_all_timepoints.pt"
Model_save_path = "trajectory_model_multistep.pth"
Batch_size = 64
Epochs = 50
learning_rate = 0.001
Hidden_dim = 512
Dropout = 0.1
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiStepTrajectoryDataset(Dataset):
    def __init__(self, pt_file):
        if not os.path.exists(pt_file):
            print(f"Error: {pt_file} not found. Run merge_embeddings.py first.")
            exit()

        self.data = torch.load(pt_file)
        print(f"Loaded {len(self.data)} full trajectory pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = item["e24"].float()

        #Stack targets as 72h, 2w, 4w into one long vector 
        y = torch.cat([item["e72"], item["e2w"], item["e4w"]]).float()
        return x, y


class MultiStepPredictor(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512):
        super().__init__()
        output_dim = input_dim * 3

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(Dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(Dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

#Cosine based loss function
class MultiStepCosineLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, pred, target):
        pred_steps = pred.view(-1, 3, 384)
        target_steps = target.view(-1, 3, 384)

        loss = 0.0
        for i in range(3):
            step_loss = 1 - self.cos(pred_steps[:, i, :], target_steps[:, i, :]).mean()
            loss += step_loss

        return loss / 3.0

#training
def train():
    print(f"Starting multi-step training on {Device}")

    full_dataset = MultiStepTrajectoryDataset(DATA_FILE)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False)

    model = MultiStepPredictor(hidden_dim=Hidden_dim).to(Device)
    criterion = MultiStepCosineLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")

    print(f"\nTraining for {Epochs} epochs...\n")

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

        avg_train_loss = running_loss / len(train_loader)

        #Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(Device), targets.to(Device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Model_save_path)
            saved_msg = " (model saved)"
        else:
            saved_msg = ""

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{Epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.6f}{saved_msg}"
        )

    print("\nTraining complete.")
    print(f"Best validation loss average cosine distance: {best_val_loss:.4f}")

    final_acc = (1 - best_val_loss) * 100
    print(f"Approximate predictive accuracy: {final_acc:.2f}%")

if __name__ == "__main__":
    train()
