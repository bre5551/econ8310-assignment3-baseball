from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import BaseballVideoDataset
from model import BaseballCNN


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            running_loss += loss.item() * X.size(0)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    root = Path(__file__).resolve().parent / "data"
    full_data = BaseballVideoDataset(data_dir=root, img_size=128, frame_step=8)

    print("Total samples:", len(full_data))
    print("Classes:", full_data.classes)

    train_size = int(0.8 * len(full_data))
    test_size = len(full_data) - train_size

    generator = torch.Generator().manual_seed(42)
    train_data, test_data = random_split(full_data, [train_size, test_size], generator=generator)

    train_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_data,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    model = BaseballCNN(num_classes=2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3

    for epoch in range(epochs):
        train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer, device)
        test_loss, test_acc = test_loop(test_loader, model, loss_fn, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    save_dir = Path(__file__).resolve().parent / "saved_models"
    save_dir.mkdir(exist_ok=True)

    save_path = save_dir / "baseball_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": full_data.classes,
            "img_size": 128,
        },
        save_path,
    )

    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()