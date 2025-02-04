import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm
from torchvision.models import resnet18, alexnet

from utils import signal_to_image
from plot_utils import (
    plot_data_distribution,
    plot_roc_curve,
    plot_loss_curve,
    plot_confusion_matrix
)

class CNNAlex(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = alexnet(pretrained=True)
        self.alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, 1)

    def forward(self, x):
        return self.alexnet(x.unsqueeze(1)).squeeze(1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return self.resnet(x.unsqueeze(1)).squeeze(1)


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 12 * 15, 128),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x).squeeze(1)


def train_and_evaluate_CNN(X, y_labels, output_folder, batch_size=32, epochs=10, class_weights=None,
                           model_name='resnet18'):
    """
    Train and evaluate a CNN model using cross-validation.

    :param X: Input data (signals).
    :param y_labels: DataFrame containing labels and metadata.
    :param output_folder: Directory to save models and plots.
    :param batch_size: Batch size for training (default: 32).
    :param epochs: Number of training epochs (default: 10).
    :param class_weights: Optional class weights for handling imbalanced data.
    :param model_name: CNN model type ('resnet18', 'alexnet', or default classifier).
    :return: DataFrame with predictions for each fold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_cms = []
    best_f1 = 0
    results_df = y_labels[['id', 'label', 'peaks']].copy()
    folds = [col for col in y_labels.columns if 'fold' in col]

    for i in range(len(folds)):

        results_df[f'pred_{i}'] = np.nan

        # Preparing dataset
        train_idx = y_labels[y_labels[f'fold_{i}'] == 'train'].x_index.to_list()
        val_idx = y_labels[y_labels[f'fold_{i}'] == 'test'].x_index.to_list()

        X_train, y_train = X[train_idx], y_labels.loc[train_idx, "label"].values
        X_val, y_val = X[val_idx], y_labels.loc[val_idx, "label"].values

        X_train_tensor = torch.tensor([signal_to_image(x) for x in X_train], dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        X_val_tensor = torch.tensor([signal_to_image(x) for x in X_val], dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Preparing weights
        if class_weights:
            weights = torch.Tensor(class_weights)
        else:
            weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
            weights = torch.tensor(weights, dtype=torch.float32, device=device)

        # Initializing model
        if model_name == 'resnet18':
            model = CNN().to(device)
        elif model_name == 'alexnet':
            model = CNNAlex().to(device)
        else:
            model = CNNClassifier().to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1] / weights[0])
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)

        train_losses, val_losses = [], []

        # Training loop
        for epoch in tqdm(range(epochs), desc=f"Training Fold {i}"):
            model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.squeeze(1))
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            train_losses.append(total_train_loss / len(train_loader))

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor.squeeze(1))
                val_losses.append(val_loss.item())

        # Evaluating model
        y_scores = torch.sigmoid(model(X_val_tensor)).cpu().detach().numpy()
        y_pred = y_scores.round()
        cm = confusion_matrix(y_val, y_pred)
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        accuracy = np.trace(cm) / np.sum(cm)
        f1 = f1_score(y_val, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(output_folder, f"best_model_fold_{i}.pth"))

        all_cms.append(cm)
        plot_confusion_matrix(cm, i, output_folder, sensitivity, specificity, accuracy)
        plot_loss_curve(train_losses, val_losses, i, output_folder)
        plot_roc_curve(y_val, y_scores, i, output_folder)
        plot_data_distribution(y_labels, i, output_folder)
        results_df.loc[val_idx, f'pred_{i}'] = y_pred.flatten()

    # Averaging across folds
    avg_cm = np.mean(all_cms, axis=0)
    sensitivity = avg_cm[1, 1] / (avg_cm[1, 0] + avg_cm[1, 1])
    specificity = avg_cm[0, 0] / (avg_cm[0, 0] + avg_cm[0, 1])
    accuracy = np.trace(avg_cm) / np.sum(avg_cm)
    plot_confusion_matrix(avg_cm, "Avg", output_folder, sensitivity, specificity, accuracy)

    return results_df
