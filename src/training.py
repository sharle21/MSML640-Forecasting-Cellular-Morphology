import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

def train(model, emb_early, emb_late, epochs=20, lr=1e-3, batch_size=64):
    dataset = TensorDataset(torch.tensor(emb_early).float(), torch.tensor(emb_late).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: MSE={total_loss/len(loader):.4f}")
    return model
