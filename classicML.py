# Classical Machine Learning: pixel-wise classification + CRF refinement

import os
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Directories
IMAGE_DIR = "Dataset/Images/"
MASK_DIR = "Dataset/Masks/"
PRED_DIR = "Results/classicalML/"
os.makedirs(PRED_DIR, exist_ok=True)

# Parameters
radius = 1
n_points = 8 * radius

def load_pair(filename):
    img_path = os.path.join(IMAGE_DIR, f"{filename}.jpg")
    mask_path = os.path.join(MASK_DIR, f"{filename}_label.PNG")
    
    image = imread(img_path)
    mask = imread(mask_path, as_gray=True) > 0.5  # Convert to binary
    return image, mask.astype(np.uint8)

def extract_features(image):
    gray = rgb2gray(image)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Flatten each channel for ML
    features = np.stack([
        gray.flatten(),
        lbp.flatten()
    ], axis=1)
    
    return features

# 1. Collect all data
X, y = [], []

for num in range(1, 119):  # 1 to 118 inclusive
    filename = f"{num:03d}"
    image, mask = load_pair(filename)
    features = extract_features(image)
    labels = mask.flatten()
    
    X.append(features)
    y.append(labels)

X = np.vstack(X)
y = np.concatenate(y)

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 4. Predict and save masks
for num in range(1, 119):
    filename = f"{num:03d}"
    image, mask = load_pair(filename)
    features = extract_features(image)
    
    prediction = clf.predict(features)
    predicted_mask = prediction.reshape(mask.shape).astype(np.uint8) * 255  # scale to 0-255

    # Save prediction
    save_path = os.path.join(PRED_DIR, f"{filename}_pred.png")
    imsave(save_path, predicted_mask)

    # Optional: visualize a few
    if num % 40 == 0:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(predicted_mask, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
