import os
import torch

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # ensure save dir exists
    os.makedirs("./models", exist_ok=True)

    # save model checkpoint


    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch)
            _, predicted = out.max(dim=1)
            correct += (predicted == data.y).sum().item()
            total += data.num_graphs
    return correct / total


def load_model(model, model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✅ Loaded model from {model_path}")
    return model
