# Classical Machine Learning: pixel-wise classification + CRF refinement

import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage.segmentation import random_walker
import matplotlib.pyplot as plt

# Paths to your dataset folders
images_dir = "Dataset/Images"
masks_dir = "Dataset/Masks"

def load_pair(filename):
    img_path = os.path.join(images_dir, filename + '.jpg')
    mask_path = os.path.join(masks_dir, filename + '_label.PNG')
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)  # Binarize mask
    return img, mask

def extract_features(img):
    # Features: intensity + Sobel X + Sobel Y edges, flattened
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    features = np.stack([img, sobelx, sobely], axis=-1)
    return features.reshape(-1, 3)  # Each pixel is a sample with 3 features

def prepare_dataset():
    all_features = []
    all_labels = []
    for num in range(1, 119):  # Assuming 118 images
        filename = f"{num:03d}"
        img, mask = load_pair(filename)
        feats = extract_features(img)
        labels = mask.flatten()
        all_features.append(feats)
        all_labels.append(labels)
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    return X, y

def train_rf(X, y):
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    return clf

def predict_and_refine(clf, img):
    feats = extract_features(img)
    pred = clf.predict(feats)
    pred_img = pred.reshape(img.shape)

    # Use random walker as a CRF alternative for refinement
    # Seeds: known crack (1) and known background (0) from prediction confidence
    # Here we generate seeds from prediction with some dilation to mark confident regions

    from scipy.ndimage import binary_dilation

    cracked = pred_img == 1
    background = pred_img == 0

    # Create markers: 1 for crack seeds, 2 for background seeds
    markers = np.zeros_like(pred_img, dtype=np.int32)
    markers[binary_dilation(cracked, iterations=2)] = 1
    markers[binary_dilation(background, iterations=2)] = 2

    refined = random_walker(img, markers, beta=130, mode='bf')
    # random_walker labels 1 and 2; map crack=1, background=2 => invert to 0/1
    refined_mask = (refined == 1).astype(np.uint8)
    return pred_img, refined_mask

def visualize(img, mask_gt, pred_mask, refined_mask, idx=0):
    plt.figure(figsize=(12, 6))
    plt.subplot(1,4,1)
    plt.title("Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask_gt, cmap='gray')
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.title("RF Prediction")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.title("Refined Mask (Random Walker)")
    plt.imshow(refined_mask, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    X, y = prepare_dataset()
    clf = train_rf(X, y)

    # Test on a single image (e.g., #5)
    test_num = 5
    filename = f"{test_num:03d}"
    img_test, mask_test = load_pair(filename)
    pred_mask, refined_mask = predict_and_refine(clf, img_test)

    visualize(img_test, mask_test, pred_mask, refined_mask)
