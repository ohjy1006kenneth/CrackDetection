import cv2
import os
import time
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize

dataset_path = "./Dataset/Images"
output_path = "Results/CrackIT_Like"
os.makedirs(output_path, exist_ok=True)

time_taken = []

for img_file in os.listdir(dataset_path):
    if img_file.endswith('.jpg') or img_file.endswith('.png'):
        img_path = os.path.join(dataset_path, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Start measure processing time
        start_time = time.time()

        # 1. Preprocessing: Stronger denoising
        blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

        # 2. Edge Detection: Only Canny
        canny = cv2.Canny(blurred, 45, 95)

        # 3. Morphological operations: Close gaps and remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4. Remove small objects (post-processing)
        closed_bool = closed.astype(bool)
        cleaned = remove_small_objects(closed_bool, min_size=100)
        cleaned = (cleaned * 255).astype(np.uint8)

        # End measure processing time
        end_time = time.time()
        time_taken.append(end_time - start_time)

        # #Visualize results
        cv2.imshow("1. Preprocessing", blurred)
        cv2.imshow("2. Canny Edge Detection", canny)
        cv2.imshow("3. Morphological Operations", closed)
        cv2.imshow("4. Cleaned Image", cleaned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save result
        result_path = os.path.join(output_path, f"crackit_{img_file}")
        cv2.imwrite(result_path, cleaned)

# Save processing time
avg_time = sum(time_taken) / len(time_taken) if time_taken else 0
with open(os.path.join(output_path, "processing_time.txt"), "w") as f:
    f.write(f"Average processing time per image: {avg_time:.4f} seconds\n")
    f.write("Individual times (seconds):\n")
    for t in time_taken:
        f.write(f"{t:.4f}\n")





        