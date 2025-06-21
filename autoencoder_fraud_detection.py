# autoencoder_fraud_detection.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('dataset/creditcard.csv')
print(f"Dataset shape: {data.shape}")

# Data preprocessing
features = data.drop(columns=['Time', 'Class'])
labels = data['Class']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

# Step 1: Initialize without training parameters
autoencoder = AutoEncoder(hidden_neuron_list=[32, 16, 32], contamination=0.001, verbose=1)

# Step 2: Set training parameters directly
autoencoder.epochs = 20
autoencoder.batch_size = 32
autoencoder.validation_size = 0.1

# Step 3: Fit the model
autoencoder.fit(X_train)

# Get predictions
y_test_scores = autoencoder.decision_function(X_test)  # The higher, the more abnormal
y_test_pred = autoencoder.predict(X_test)              # 0: normal, 1: outlier


# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

auc = roc_auc_score(y_test, y_test_scores)
print(f"ROC AUC Score: {auc:.4f}")

# Plot reconstruction error
plt.figure(figsize=(8, 5))
plt.hist(y_test_scores[y_test == 0], bins=50, alpha=0.6, label='Normal')
plt.hist(y_test_scores[y_test == 1], bins=50, alpha=0.6, label='Fraud')
plt.title('Reconstruction Error')
plt.xlabel('Error Score')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig("results/output.png")
plt.show()
