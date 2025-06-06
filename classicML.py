# Classical Machine Learning: pixel-wise classification + CRF refinement

import os
import numpy as np
import time
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from skimage.morphology import remove_small_objects
from skimage.morphology import opening, closing, disk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
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

def extract_features_and_labels(image, mask):
    features = extract_features(image)
    labels = mask.flatten()

    # Balance classes
    crack_idx = np.where(labels == 1)[0]
    non_crack_idx = np.where(labels == 0)[0]

    if len(crack_idx) == 0:
        return np.empty((0, features.shape[1])), np.empty((0,), dtype=np.uint8)

    non_crack_sampled = resample(non_crack_idx,
                                replace=False,
                                n_samples=len(crack_idx),
                                random_state=42)

    balanced_idx = np.concatenate([crack_idx, non_crack_sampled])
    np.random.shuffle(balanced_idx)

    return features[balanced_idx], labels[balanced_idx]

def extract_features(image):
    gray = (rgb2gray(image) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    edges = sobel(gray)
    blur = gaussian(gray, sigma=1)  # smooth intensity
    blur = (blur * 255).astype(np.uint8)
    
    h, w = gray.shape
    # normalized coordinates
    coords_x, coords_y = np.meshgrid(np.arange(w), np.arange(h))
    coords_x = coords_x.flatten() / w
    coords_y = coords_y.flatten() / h
    
    features = np.stack([
        gray.flatten(),
        lbp.flatten(),
        edges.flatten(),
        blur.flatten(),
        coords_x,
        coords_y
    ], axis=1)
    return features

def clean_prediction(predicted_mask, min_size=400):
    binary_mask = predicted_mask.astype(bool)
    cleaned = remove_small_objects(binary_mask, min_size=min_size)
    cleaned = closing(cleaned, disk(3))  # close small gaps in cracks
    cleaned = opening(cleaned, disk(2))  # remove tiny noise
    return (cleaned.astype(np.uint8) * 255)

# List of all filenames (strings like '001', '002', ..., '118')
all_filenames = [f"{num:03d}" for num in range(1, 119)]

# Split filenames into train and test
train_files, test_files = train_test_split(all_filenames, test_size=0.2, random_state=42)

print(f"Training on {len(train_files)} images: {train_files}")
print(f"Testing on {len(test_files)} images: {test_files}")

# Collect training data
X_train_list, y_train_list = [], []

for filename in train_files:
    image, mask = load_pair(filename)
    features, labels = extract_features_and_labels(image, mask)
    if features.shape[0] > 0:
        X_train_list.append(features)
        y_train_list.append(labels)

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)


# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Predict on test set
time_taken = []

for filename in test_files:
    start_time = time.time()

    image, mask = load_pair(filename)
    features = extract_features(image)
    
    proba = clf.predict_proba(features)[:, 1]  # probability for crack class
    threshold = 0.9  # tweak this threshold between 0.5-0.8 based on validation
    prediction = (proba > threshold).astype(np.uint8)
    predicted_mask = prediction.reshape(mask.shape) * 255
    predicted_mask = clean_prediction(predicted_mask)

    end_time = time.time()
    time_taken.append(end_time - start_time)

    # Save prediction
    save_path = os.path.join(PRED_DIR, f"classML_{filename}.jpg")
    imsave(save_path, predicted_mask)

    # Optional: visualize a few
    if int(filename) % 40 == 0:
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

# Save processing time
avg_time = sum(time_taken) / len(time_taken) if time_taken else 0
with open(os.path.join(PRED_DIR, "processing_time.txt"), "w") as f:
    f.write(f"Average processing time per test image: {avg_time:.4f} seconds\n")
    f.write("Individual times (seconds):\n")
    for t in time_taken:
        f.write(f"{t:.4f}\n")
