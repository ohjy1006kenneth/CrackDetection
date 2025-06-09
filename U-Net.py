import os
import time
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set dataset paths
IMAGE_DIR = "Dataset/Images"
MASK_DIR = "Dataset/Masks"
OUTPUT_DIR = "Results/U-Net"
MODEL_PATH = "models/unet_crack_detector.keras"

time_taken = []

# Load and preprocess data without resizing
def load_data(image_dir, mask_dir):
    image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.PNG')))

    images = [cv2.imread(img_path) / 255.0 for img_path in image_paths]
    masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0 for mask_path in mask_paths]

    images = np.array(images, dtype=np.float32)
    masks = np.expand_dims(np.array(masks, dtype=np.float32), axis=-1)
    return images, masks

# Build U-Net dynamically based on input shape
def build_unet(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    # Bottleneck
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)

    # Decoder
    u5 = UpSampling2D()(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = UpSampling2D()(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = UpSampling2D()(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    outputs = Conv2D(1, 1, activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    return model

# Load data
images, masks = load_data(IMAGE_DIR, MASK_DIR)

# Split data
indices = np.arange(len(images))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
x_train, x_test = images[train_idx], images[test_idx]
y_train, y_test = masks[train_idx], masks[test_idx]

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
]

# Check for saved model
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("No saved model found, building a new one...")
    input_shape = x_train.shape[1:]  # dynamic input shape
    model = build_unet(input_shape)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=4,
        validation_split=0.1,
        callbacks=callbacks
    )
    print("Model training complete.")
    model.save(MODEL_PATH)

# Predict
start_time = time.time()
preds = model.predict(x_test)
preds = (preds > 0.5).astype(np.uint8)
end_time = time.time()
time_taken.append(end_time - start_time)

# Save predictions
os.makedirs(OUTPUT_DIR, exist_ok=True)
for i, pred in enumerate(preds):
    test_image_index = test_idx[i]
    filename = f"ML_{(test_image_index+1):03d}.png"  # Zero-padded to 3 digits
    pred_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(pred_path, (pred.squeeze() * 255))

print("Prediction complete. Check the 'Results/U-Net' folder.")

# Save processing time
avg_time = sum(time_taken) / len(time_taken) if time_taken else 0
with open(os.path.join(OUTPUT_DIR, "processing_time.txt"), "w") as f:
    f.write(f"Average processing time per image: {avg_time:.4f} seconds\n")
    f.write("Individual times (seconds):\n")
    for t in time_taken:
        f.write(f"{t:.4f}\n")
