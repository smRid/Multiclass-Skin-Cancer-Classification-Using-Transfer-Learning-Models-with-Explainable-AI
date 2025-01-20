import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.api.models import load_model
from keras.api.preprocessing.image import load_img, img_to_array
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)

# --------------------------
# PATHS AND MODEL LOADING
# --------------------------

# Paths
model_path = r"E:\Mehedy\Model\Model File\VGG19_Model_1.h5"
image_dir = r"test_image"
csv_path = r"Dataset\test_set.csv"

# Load model
model = load_model(model_path)

# Load CSV
test_set = pd.read_csv(csv_path)

# Extract image paths and labels
test_set["image_path"] = test_set["image_id"] + ".jpg"
test_set["image_full_path"] = test_set["image_path"].apply(
    lambda x: os.path.join(image_dir, x)
)
labels = test_set["label"].values

# --------------------------
# IMAGE PREPROCESSING
# --------------------------

# Image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    """Loads and preprocesses an image."""
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0
    return image

# Load and preprocess images
images = np.array(
    [
        preprocess_image(img_path)
        for img_path in test_set["image_full_path"]
        if os.path.exists(img_path)
    ]
)
labels = np.array(
    [
        labels[i]
        for i, img_path in enumerate(test_set["image_full_path"])
        if os.path.exists(img_path)
    ]
)

# --------------------------
# PREDICTIONS
# --------------------------

# Predict
predictions = model.predict(images)
predicted_classes = np.argmax(predictions, axis=1)

# --------------------------
# METRICS CALCULATION
# --------------------------

# Metrics calculation
accuracy = accuracy_score(labels, predicted_classes)
f1 = f1_score(labels, predicted_classes, average="weighted")
precision = precision_score(labels, predicted_classes, average="weighted")
recall = recall_score(labels, predicted_classes, average="weighted")
auc_roc = roc_auc_score(labels, predictions, multi_class="ovr")

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

# --------------------------
# CALCULATE SPECIFICITY
# --------------------------

# Initialize variables for specificity calculation
specificity_per_class = []
for i in range(len(np.unique(labels))):
    # For each class, create a binary confusion matrix
    # Positive class is `i`, negative class is everything else
    binary_labels = (labels == i).astype(int)
    binary_predictions = (predicted_classes == i).astype(int)
    tn, fp, fn, tp = confusion_matrix(binary_labels, binary_predictions).ravel()
    
    # Specificity formula: TN / (TN + FP)
    specificity = tn / (tn + fp)
    specificity_per_class.append(specificity)

# Average specificity across all classes
average_specificity = np.mean(specificity_per_class)

# Print specificity for each class and average specificity
for i, spec in enumerate(specificity_per_class):
    print(f"Specificity for Class {i}: {spec:.4f}")
print(f"Average Specificity: {average_specificity:.4f}")

# --------------------------
# METRICS VISUALIZATION (UPDATED)
# --------------------------

# Update metrics to include average specificity
metrics = {
    "Accuracy": accuracy,
    "F1 Score": f1,
    "Precision": precision,
    "Recall": recall,
    "Specificity": average_specificity,
    "AUC-ROC": auc_roc,
    # "Specificity": average_specificity
}

# Bar plot for metrics
plt.figure(figsize=(10, 5))
plt.bar(metrics.keys(), metrics.values(), color="skyblue")
plt.title("Model Evaluation Metrics")
plt.ylabel("Scores")
plt.ylim(0, 1)
plt.show()

# --------------------------
# ROC CURVE PLOT
# --------------------------

# Plot ROC curve (one vs rest)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(predictions.shape[1]):
    fpr[i], tpr[i], _ = roc_curve((labels == i).astype(int), predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure()
for i in range(predictions.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# --------------------------
# CONFUSION MATRIX
# --------------------------

# Confusion Matrix
conf_matrix = confusion_matrix(labels, predicted_classes)

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=np.unique(labels),
    yticklabels=np.unique(labels),
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
