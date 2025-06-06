import os
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

# Paths
pred_path = "Results/classicalML"
gt_path = "Dataset/Masks"

# Get list of prediction files
pred_files = sorted([f for f in os.listdir(pred_path) if f.endswith('.jpg')])

# Initialize metric lists
# Precision: The porportion of pixels predicted as crack that are actually cracks.
# Recall: The proportion of actual cracks that were correctly predicted.
# F1 Score: The harmonic mean of precision and recall.
# Jaccard Index (IoU): The intersection over union of predicted and ground truth masks.
precisions, recalls, f1s, jaccards = [], [], [], []

for fname in pred_files:
    # Extract the numeric part from the prediction filename
    num = fname.split('_', 1)[-1].rsplit('.', 1)[0]
    gt_fname = f"{num}_label.PNG"
    pred_file = os.path.join(pred_path, fname)
    gt_file = os.path.join(gt_path, gt_fname)
    if not os.path.exists(gt_file):
        print(f"Ground truth for {fname} not found (expected {gt_fname}), skipping.")
        continue

    # Read images as grayscale
    pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)

    # Binarize (assuming cracks are white, background is black)
    _, pred_bin = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)
    _, gt_bin = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)

    # Flatten for sklearn metrics
    pred_flat = pred_bin.flatten()
    gt_flat = gt_bin.flatten()

    # Compute metrics
    precisions.append(precision_score(gt_flat, pred_flat, zero_division=0))
    recalls.append(recall_score(gt_flat, pred_flat, zero_division=0))
    f1s.append(f1_score(gt_flat, pred_flat, zero_division=0))
    jaccards.append(jaccard_score(gt_flat, pred_flat, zero_division=0))

     # Visualization of errors
    # Convert ground truth to 3-channel for visualization
    gt_vis = cv2.cvtColor((gt_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Find error and correct pixels
    error_mask = (gt_bin != pred_bin)
    correct_mask = (gt_bin == pred_bin)

    # Only consider white lines (cracks) on the ground truth
    crack_mask = (gt_bin == 1)
    correct_mask = crack_mask & (pred_bin == 1)  # True positive: crack detected
    error_mask = crack_mask & (pred_bin == 0)    # False negative: crack missed

    # Draw correct in green, errors in red
    gt_vis[correct_mask] = [0, 255, 0]  # Green for correct crack detection
    gt_vis[error_mask] = [0, 0, 255]    # Red for missed cracks

    # Save visualization
    out_vis_path = os.path.join(pred_path, "ErrorViz")
    os.makedirs(out_vis_path, exist_ok=True)
    cv2.imwrite(os.path.join(out_vis_path, f"vis_{num}.png"), gt_vis)

# Print average metrics
precision_avg = np.mean(precisions)
recall_avg = np.mean(recalls)
f1_avg = np.mean(f1s)
iou_avg = np.mean(jaccards)

print(f"Precision: {precision_avg:.4f}")
print(f"Recall:    {recall_avg:.4f}")
print(f"F1 Score:  {f1_avg:.4f}")
print(f"IoU:       {iou_avg:.4f}")

# Write average metrics to a text file
with open(os.path.join(pred_path, "benchmark_results.txt"), "w") as f:
    f.write(f"Precision: {precision_avg:.4f}\n")
    f.write(f"Recall:    {recall_avg:.4f}\n")
    f.write(f"F1 Score:  {f1_avg:.4f}\n")
    f.write(f"IoU:       {iou_avg:.4f}\n")