import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from DataLoader import ECGDataLoader

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.model(x)


def l2_regularization(model, lambda_l2):
    """ Compute L2 regularization term for model parameters. """
    l2_norm = sum(torch.norm(param, p=2) ** 2 for param in model.parameters())
    return lambda_l2 * l2_norm


def plot_confusion_matrix(cm, fold, output_folder):
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Not Noisy", "Noisy"],
                yticklabels=["Not Noisy", "Noisy"], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - Fold {fold}")

    save_and_close_plot(fig, output_folder, f"confusion_matrix_fold_{fold}.png")


def plot_loss_curve(train_losses, val_losses, fold, output_folder):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_losses, label="Training Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training vs Validation Loss - Fold {fold}")
    ax.legend()

    save_and_close_plot(fig, output_folder, "loss_curve.png")


def train_and_evaluate(X, y_labels, output_folder, batch_size=32, epochs=10, l2_lambda=1e-4):
    input_dim = X.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_cms = []

    for i in range(5):  # cv_k = 5 folds
        train_idx = y_labels[y_labels[f'fold_{i}'] == 'train'].x_index.to_list()
        val_idx = y_labels[y_labels[f'fold_{i}'] == 'test'].x_index.to_list()

        X_train, y_train = X[train_idx], y_labels.loc[train_idx, "label"].values
        X_val, y_val = X[val_idx], y_labels.loc[val_idx, "label"].values

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = MLPClassifier(input_dim).to(device)

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        pos_weight = len(y_train) / (2*pos_count)
        neg_weight = len(y_train) / (4*neg_count)

        class_weights = torch.tensor([pos_weight / neg_weight], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, val_losses = [], []

        # Training loop
        for epoch in tqdm(range(epochs), f"Epoches training for fold {i}"):
            model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Add L2 regularization
                loss += l2_regularization(model, l2_lambda)

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_losses.append(val_loss.item())

        # Compute Confusion Matrix
        model.eval()
        with torch.no_grad():
            y_pred = torch.sigmoid(model(X_val_tensor)).cpu().numpy().round()

        cm = confusion_matrix(y_val, y_pred)
        all_cms.append(cm)

        # Plot CM and Loss Curve for this fold
        plot_confusion_matrix(cm, i, output_folder)
        plot_loss_curve(train_losses, val_losses, i, output_folder)

    avg_cm = np.mean(all_cms, axis=0)
    print("Average Confusion Matrix:")
    plot_confusion_matrix(avg_cm, "Avg", output_folder)

def save_and_close_plot(fig, output_folder, filename, timeout=5):
    """ Save the plot and close it automatically after timeout seconds """
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, filename))
    plt.pause(timeout)  # Display for a few seconds
    plt.close(fig)

# Load data
loader = ECGDataLoader()

X, y_labels = loader.read_arrays('output/X_array.npy', 'output/y_label_df.csv')

# Train and evaluate
output_folder = 'output'
train_and_evaluate(X, y_labels, output_folder)
