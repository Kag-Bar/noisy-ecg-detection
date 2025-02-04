import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from utils.utils import signal_to_image
from utils.plot_utils import plot_confusion_matrix
from train_and_eval_CNN import CNN, CNNAlex, CNNClassifier

def test_and_eval(X, y_labels, model_path, model_name='resnet18', test=True, output_folder=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if test:
        val_idx = y_labels.x_index.to_list()
    else:
        fold = int(model_path.split('_')[-1].split('.')[0])
        val_idx = y_labels[y_labels[f'fold_{fold}'] == 'test'].x_index.to_list()

    X_val, y_val = X[val_idx], y_labels.loc[val_idx, "label"].values

    # Initializing model
    if model_name == 'resnet18':
        model = CNN().to(device)
    elif model_name == 'alexnet':
        model = CNNAlex().to(device)
    else:
        model = CNNClassifier().to(device)

    # Loading model
    if device == 'cpu':
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))

    model.eval()

    X_val_preds = torch.sigmoid(model(torch.tensor([signal_to_image(x) for x in X_val], dtype=torch.float32))).detach().numpy().round()

    cm = confusion_matrix(y_val, X_val_preds)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    accuracy = np.trace(cm) / np.sum(cm)
    plot_confusion_matrix(cm, 'test', output_folder, sensitivity, specificity, accuracy)

    return X_val_preds