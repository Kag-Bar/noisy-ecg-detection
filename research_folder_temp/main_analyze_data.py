import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

from DataLoader import ECGDataLoader

# Usage
loader = ECGDataLoader()
loader.load_data(
    folder_path='../set-a',
    labels_path=['set-a\RECORDS-acceptable', 'set-a\RECORDS-unacceptable'],
    data_type='dat',
    origin_freq=500,
    normalize=True,
    resample_freq=True,
    augment=False)

loader.load_data(
    folder_path='../training2017',
    labels_path=['training2017/REFERENCE-original.csv'],
    data_type='mat',
    origin_freq=300,
    normalize=True,
    resample_freq=False,
    augment=True)


# Access processed data
X, y_labels = loader.finalize_data()
y_labels = loader.train_test_split(y_labels,True, 0.4)
y_labels = loader.analyze(y_labels, output_folder='output/HPF')
loader.save_arrays(X, y_labels, 'output/HPF')

#
choosen_fold = 'fold_0'

X, df = loader.read_arrays('output_cnn/X_array_0.4_noise_ratio.npy', 'output_cnn/y_label_df_0.4_noise_ratio.csv')

# Load Data
df = df.dropna(subset=[choosen_fold])

# Extract train and test sets
train_df = df[df[choosen_fold] == "train"]
test_df = df[df[choosen_fold] == "test"]

X_train, y_train = train_df["peaks"].values.reshape(-1, 1), train_df["label"].values
X_test, y_test = test_df["peaks"].values.reshape(-1, 1), test_df["label"].values

# --- 1. Logistic Regression ---
log_reg = LogisticRegression(penalty=None, class_weight='balanced', solver='newton-cholesky')
log_reg.fit(X_train, y_train)
log_preds = log_reg.predict(X_test)

# --- 2. RF Model ---
rf_clf = RandomForestClassifier(n_estimators=50, criterion='log_loss', class_weight='balanced_subsample', random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)

# --- 3. Gaussian Mixture Model (GMM) ---
gmm = GaussianMixture(n_components=2)
gmm.fit(X_train)
gmm_preds = gmm.predict(X_test)

# --- Plot Confusion Matrices ---
models = {
    "Logistic Regression": log_preds,
    "RF": rf_pred,
    "GMM": gmm_preds
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()

for i, (name, preds) in enumerate(models.items()):
    cm = confusion_matrix(y_test, preds)

    # Compute metrics
    accuracy = np.trace(cm) / np.sum(cm)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Valid", "Noisy"], yticklabels=["Valid", "Noisy"],
                ax=axes[i])

    # Update title with metrics
    axes[i].set_title(f"{name}\nAcc: {accuracy:.2f}, Sens: {sensitivity:.2f}, Spec: {specificity:.2f}")
    axes[i].set_xlabel("Predicted Label")
    axes[i].set_ylabel("Actual Label")

plt.tight_layout()
plt.show()