import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utils import signal_to_image

def plot_signal_heatmap(signal):
    img = signal_to_image(signal)
    plt.figure(figsize=(6, 6))
    sns.heatmap(img, cmap='coolwarm')
    plt.title("ECG Signal Heatmap")
    plt.show()

def plot_confusion_matrix(cm, fold, output_folder, sensitivity, specificity, accuracy):
    plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Not Noisy", "Noisy"],
                yticklabels=["Not Noisy", "Noisy"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    if fold == 'test':
        title = "Results on Test\nSens: {sensitivity:.2f}, Spec: {specificity:.2f}, Acc: {accuracy:.2f}"
    else:
        title = f"Confusion Matrix - Fold {fold}\nSens: {sensitivity:.2f}, Spec: {specificity:.2f}, Acc: {accuracy:.2f}"
    plt.title(title)
    if output_folder:
        plt.savefig(os.path.join(output_folder, f"{title}.png"))
    plt.show()

def plot_loss_curve(train_losses, val_losses, fold, output_folder):
    plt.subplots(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    title = f"Training vs Validation Loss - Fold {fold}"
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"{title}.png"))
    plt.show()


def plot_roc_curve(y_true, y_scores, fold, output_folder):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {fold}')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"ROC_Curve_Fold_{fold}_AUC_{roc_auc}.png"))
    plt.show()

def plot_data_distribution(y_labels, fold, output_folder):
    plt.figure()
    sns.histplot(y_labels[y_labels[f'fold_{fold}'] == 'train']["label"], label="Train", kde=True, color='blue', alpha=0.5)
    sns.histplot(y_labels[y_labels[f'fold_{fold}'] == 'test']["label"], label="Validation", kde=True, color='red', alpha=0.5)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title(f"Data Distribution - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"Data_Distribution_Fold_{fold}.png"))
    plt.show()

def plot_FP_FN_signals(X, y_labels, model_path, X_val_preds, test=False):
    fold = int(model_path.split('_')[-1].split('.')[0])
    if test:
        val_idx = y_labels.x_index.to_list()
    else:
        val_idx = y_labels[y_labels[f'fold_{fold}'] == 'test'].x_index.to_list()
    ids_val = y_labels.loc[val_idx, "id"].values  # Get IDs for labeling
    X_val, y_val = X[val_idx], y_labels.loc[val_idx, "label"].values

    # Identify FP and FN indices
    fp_indices = [i for i in range(len(y_val)) if y_val[i] == 0 and X_val_preds[i] == 1]  # False Positives
    fn_indices = [i for i in range(len(y_val)) if y_val[i] == 1 and X_val_preds[i] == 0]  # False Negatives

    # Sample up to 10 examples from each group
    fp_samples = np.random.choice(fp_indices, size=min(10, len(fp_indices)), replace=False) if fp_indices else []
    fn_samples = np.random.choice(fn_indices, size=min(10, len(fn_indices)), replace=False) if fn_indices else []

    # Create subplots for FP
    if len(fp_samples) > 0:
        fig, axes = plt.subplots(5, 2, figsize=(10, 10))
        fig.suptitle("False Positives (Predicted Noisy, Actually Valid)", fontsize=14)

        for i, idx in enumerate(fp_samples):
            row, col = divmod(i, 2)
            axes[row, col].plot(X_val[idx])
            axes[row, col].set_title(f"ID: {ids_val[idx]}", fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # Create subplots for FN
    if len(fn_samples) > 0:
        fig, axes = plt.subplots(5, 2, figsize=(10, 10))
        fig.suptitle("False Negatives (Predicted Valid, Actually Noisy)", fontsize=14)

        for i, idx in enumerate(fn_samples):
            row, col = divmod(i, 2)
            axes[row, col].plot(X_val[idx])
            axes[row, col].set_title(f"ID: {ids_val[idx]}", fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()