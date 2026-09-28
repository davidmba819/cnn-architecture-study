
# ============================================================
# CNN Architecture Study
# Reusable utilities for the entire project
# ============================================================

import random
import numpy as np
import torch

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
  """Set random seeds for reproducible experiments."""

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Data Preparation
# ============================================================

from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def create_transforms(image_size=(224,224)):

  """
    Create image transformations for training and evaluation.

    Training:
        - Resize
        - Random horizontal flip
        - Convert to tensor
        - Normalize

    Validation/Test:
        - Resize
        - Convert to tensor
        - Normalize
    """

  train_transform = transforms.Compose([
      transforms.Resize(image_size),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225]
      )
  ])

  eval_transform = transforms.Compose([
      transforms.Resize(image_size),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225]
      )
  ])

  return train_transform, eval_transform

# ============================================================
# Dataset Preparation
# ============================================================

from torch.utils.data import random_split

def prepare_datasets(dataset, val_ratio=0.2, seed=42):

  """
    Split the original training dataset into training and
    validation sets while keeping the original test set separate.

    Parameters
    ----------
    dataset : Hugging Face DatasetDict
        Stanford Cars dataset containing 'train' and 'test'.

    val_ratio : float
        Proportion of the original training data used for validation.

    seed : int
        Random seed used for reproducible splitting.

    Returns
    -------
    train_dataset
    val_dataset
    test_dataset
    """

  full_train = dataset["train"]
  test_dataset = dataset["test"]

  train_size = int(len(full_train) * (1 - val_ratio))
  val_size = len(full_train) - train_size

  generator = torch.Generator().manual_seed(seed)

  train_dataset, val_dataset = random_split(
      full_train,
      [train_size, val_size],
      generator=generator
  )

  return train_dataset, val_dataset, test_dataset

# ============================================================
# DataLoader Creation
# ============================================================

from torch.utils.data import Dataset, DataLoader

def create_dataloader(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=64,
    num_workers=2
):

  """
    Apply transformations and create PyTorch DataLoaders.

    Parameters
    ----------
    train_dataset : Dataset
        Training dataset.

    val_dataset : Dataset
        Validation dataset.

    test_dataset : Dataset
        Test dataset.

    train_transform : torchvision.transforms.Compose
        Transformations applied to training images.

    eval_transform : torchvision.transforms.Compose
        Transformations applied to validation and test images.

    batch_size : int
        Number of images processed in each batch.

    num_workers : int
        Number of worker processes used by DataLoader.

    Returns
    -------
    train_loader
    val_loader
    test_loader
    """

  class TransformedDataset(Dataset):
      def __init__(self, dataset, transform=None):
          self.dataset = dataset
          self.transform = transform

      def __len__(self):
        return len(self.dataset)


      def __getitem__(self, index):

        item = self.dataset[index]
        image = item["image"]
        label = item["label"]

        if self.transform:
          image = self.transform(image)

        return image, label

  train_dataset = TransformedDataset(train_dataset, transform=train_transform)

  val_dataset = TransformedDataset(val_dataset, transform=eval_transform)

  test_dataset = TransformedDataset(test_dataset, transform=eval_transform)

  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers
  )

  val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers
  )

  test_loader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers
  )

  return train_loader, val_loader, test_loader
  

# ============================================================
# Model Training
# ============================================================

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):

  """
    Train a model and evaluate it on the validation set
    after every epoch.

    Returns
    -------
    model : trained model
    history : dictionary containing loss and accuracy
    """

  history = {
      "train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": []
  }

  model.to(device)

  for epoch in range(epochs):

    model.train()

    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0

    for images, labels in train_loader:

      images = images.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      outputs = model(images)

      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * images.size(0)

      prediction = torch.argmax(outputs, dim=1)

      train_correct += (prediction == labels).sum().item()
      train_total += labels.size(0)

    train_loss = running_loss / train_total
    train_acc = train_correct / train_total

    # ----------------------------------------------------
    # Validation
    # ----------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0.0
    val_total = 0

    with torch.no_grad():

      for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        loss = loss_fn(output, labels)

        val_loss += loss.item() * images.size(0)

        prediction = torch.argmax(output, dim=1)

        val_correct += (prediction == labels).sum().item()
        val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

  return model, history

# ============================================================
# Training History Visualization
# ============================================================

import matplotlib.pyplot as plt


def plot_history(history):
    """
    Plot training and validation loss and accuracy.

    Parameters
    ----------
    history : dict
        History dictionary returned by train_model().
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

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

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_running_loss / test_total
    test_acc = test_correct / test_total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return test_loss, test_acc

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded from: {path}")

    return model
from datasets import load_from_disk

def load_dataset(data_path):
    """
    Load the Stanford Cars dataset from disk.

    Parameters
    ----------
    data_path : str
        Path to the saved dataset.

    Returns
    -------
    DatasetDict
        Loaded Stanford Cars dataset.
    """
    dataset = load_from_disk(data_path)

    print("Dataset loaded successfully.")
    print(dataset)

    return dataset
def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    train_transform,
    eval_transform,
    batch_size=64,
    num_workers=2
):

    """
    Apply transformations and create PyTorch DataLoaders.

    Parameters
    ----------
    train_dataset : Dataset
        Training dataset.

    val_dataset : Dataset
        Validation dataset.

    test_dataset : Dataset
        Test dataset.

    train_transform : torchvision.transforms.Compose
        Transformations applied to training images.

    eval_transform : torchvision.transforms.Compose
        Transformations applied to validation and test images.

    batch_size : int
        Number of images processed in each batch.

    num_workers : int
        Number of worker processes used by DataLoader.

    Returns
    -------
    train_loader
    val_loader
    test_loader
    """

    class TransformedDataset(Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            item = self.dataset[index]
            image = item["image"]
            label = item["label"]

            if self.transform:
                image = self.transform(image)

            return image, label

    train_dataset = TransformedDataset(
        train_dataset,
        transform=train_transform
    )

    val_dataset = TransformedDataset(
        val_dataset,
        transform=eval_transform
    )

    test_dataset = TransformedDataset(
        test_dataset,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):

  """
    Train a model and evaluate it on the validation set
    after every epoch.

    Returns
    -------
    model : trained model
    history : dictionary containing loss and accuracy
    """

  history = {
      "train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": []
  }

  model.to(device)

  for epoch in range(epochs):

    model.train()

    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0

    for images, labels in train_loader:

      images = images.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      outputs = model(images)

      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

      train_loss += loss.item() * images.size(0)

      prediction = torch.argmax(outputs, dim=1)

      train_correct += (prediction == labels).sum().item()
      train_total += labels.size(0)

    train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    # ----------------------------------------------------
    # Validation
    # ----------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0.0
    val_total = 0

    with torch.no_grad():

      for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        loss = loss_fn(output, labels)

        val_loss += loss.item() * images.size(0)

        prediction = torch.argmax(output, dim=1)

        val_correct += (prediction == labels).sum().item()
        val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

  return model, history

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):

  """
    Train a model and evaluate it on the validation set
    after every epoch.

    Returns
    -------
    model : trained model
    history : dictionary containing loss and accuracy
    """

  history = {
      "train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": []
  }

  model.to(device)

  for epoch in range(epochs):

    # ----------------------------------------------------
    # Training
    # ----------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

      images = images.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      outputs = model(images)

      loss = loss_fn(outputs, labels)

      loss.backward()

      optimizer.step()

      train_loss += loss.item() * images.size(0)

      prediction = torch.argmax(outputs, dim=1)

      train_correct += (prediction == labels).sum().item()
      train_total += labels.size(0)

    train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    # ----------------------------------------------------
    # Validation
    # ----------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

      for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        val_loss += loss.item() * images.size(0)

        prediction = torch.argmax(outputs, dim=1)

        val_correct += (prediction == labels).sum().item()
        val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    # ----------------------------------------------------
    # Save history
    # ----------------------------------------------------

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)

    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

  return model, history
