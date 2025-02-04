import os
import torch
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from plot_utils import plot_confusion_matrix, plot_loss_curve

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def train_and_evaluate(X, y_labels, output_folder, batch_size=32, epochs=20, l2_lambda=5e-3):
    input_dim = X.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_cms = []

    for i in range(5):
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
        criterion = FocalLoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=l2_lambda)

        train_losses, val_losses = [], []

        for epoch in tqdm(range(epochs), f"Epoch Training for fold {i}"):
            model.train()
            total_train_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_losses.append(val_loss.item())

        model.eval()
        with torch.no_grad():
            y_pred = torch.sigmoid(model(X_val_tensor)).cpu().numpy().round()

        cm = confusion_matrix(y_val, y_pred)
        all_cms.append(cm)
        plot_confusion_matrix(cm, i, output_folder)
        plot_loss_curve(train_losses, val_losses, i, output_folder)

    avg_cm = np.mean(all_cms, axis=0)
    plot_confusion_matrix(avg_cm, "Avg", output_folder)
