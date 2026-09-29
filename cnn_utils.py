# ============================================================
# CNN Architecture Study
# Reusable utilities for the entire project
# ============================================================

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data Preparation
# ============================================================

def create_transforms():

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

    return train_transform, eval_transform


def prepare_datasets(train_dataset, val_ratio=0.2, seed=42):

    train_size = int((1 - val_ratio) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    generator = torch.Generator().manual_seed(seed)

    train_data, val_data = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )

    return train_data, val_data


# ============================================================
# DataLoader Creation
# ============================================================

def create_dataloaders(
    train_data,
    val_data,
    test_data,
    train_transform,
    eval_transform,
    batch_size=64,
    num_workers=2
):

    class TransformedDataset(Dataset):

        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            image, label = self.dataset[index]

            if self.transform:
                image = self.transform(image)

            return image, label

    train_transformed = TransformedDataset(
        train_data,
        transform=train_transform
    )

    val_transformed = TransformedDataset(
        val_data,
        transform=eval_transform
    )

    test_transformed = TransformedDataset(
        test_data,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_transformed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_transformed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_transformed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Model Training
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    device,
    epochs
):

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    model.to(device)

    for epoch in range(epochs):

        # Training
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = loss_function(outputs, labels)

                val_running_loss += loss.item() * images.size(0)

                predicted = torch.argmax(outputs, dim=1)

                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return model, history


# ============================================================
# Training History
# ============================================================

def plot_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_acc"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# Model Evaluation
# ============================================================

def test_model(
    model,
    test_loader,
    loss_function,
    device
):

    model.to(device)
    model.eval()

    test_running_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_function(outputs, labels)

            test_running_loss += loss.item() * images.size(0)

            predicted = torch.argmax(outputs, dim=1)

            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_loss = test_running_loss / test_total
    test_acc = test_correct / test_total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return test_loss, test_acc


# ============================================================
# Model Saving and Loading
# ============================================


def save_model(model, path):

    torch.save(model.state_dict(), path)

    print(f"Model saved to: {path}")


def load_model(model, path, device):

    model.load_state_dict(
        torch.load(path, map_location=device)
    )

    model.to(device)
    model.eval()

    print(f"Model loaded from: {path}")

    return model

