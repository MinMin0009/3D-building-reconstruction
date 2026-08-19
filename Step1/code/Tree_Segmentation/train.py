import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from data import normalize_images
from model import m_unet
import cv2
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

# Paths to the image and mask directories
image_dir = r"D:\project\suseou_data3\Images_processing\data\images"
mask_dir = r"D:\project\suseou_data3\Images_processing\data\masks"

# Hyperparameters
input_size = (256, 256, 3)  # Input size for the model
learning_rate = 1e-4        # Learning rate for the optimizer
dropout_rate = 0.5          # Dropout rate
batch_size = 8              # Batch size for training
epochs = 200                # Number of training epochs
validation_split = 0.2      # Fraction of data for validation

# Load and preprocess images and masks
def load_images_and_masks(image_dir, mask_dir, target_size=(256, 256)):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    images = []
    masks = []

    for image_file, mask_file in zip(image_files, mask_files):
        image = cv2.imread(os.path.join(image_dir, image_file))
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Error: Could not read {image_file} or {mask_file}")
            continue

        # Resize images and masks to target size
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).astype(np.uint8)

        images.append(image)
        masks.append(mask)

    print(f"Loaded {len(images)} images and {len(masks)} masks.")
    return np.array(images), np.array(masks)

# Dice Coefficient metric
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Chuyển đổi y_true sang kiểu float32
    y_pred = tf.round(y_pred)  # Làm tròn y_pred về 0 hoặc 1
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon())

# Main script
print("Loading and preprocessing data...")
images, masks = load_images_and_masks(image_dir, mask_dir)
masks = np.expand_dims(masks, axis=-1)  # Add channel dimension for masks

# Normalize images for training
images = normalize_images(images)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=validation_split, random_state=42)

# Custom callback for visualizing predictions and metrics
class VisualizeCallback(Callback):
    def __init__(self, X_train, y_train, X_val, y_val, interval=10):
        super(VisualizeCallback, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            # Visualize metrics
            self._plot_metrics()

            # Visualize predictions
            self._plot_predictions(epoch)

    def _plot_metrics(self):
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        ax[0].plot(self.model.history.history['loss'], label='Train Loss')
        ax[0].plot(self.model.history.history['val_loss'], label='Val Loss')
        ax[0].set_title('Loss')
        ax[0].legend()

        # Precision
        ax[1].plot(self.model.history.history['precision'], label='Train Precision')
        ax[1].plot(self.model.history.history['val_precision'], label='Val Precision')
        ax[1].set_title('Precision')
        ax[1].legend()

        # Recall
        ax[2].plot(self.model.history.history['recall'], label='Train Recall')
        ax[2].plot(self.model.history.history['val_recall'], label='Val Recall')
        ax[2].set_title('Recall')
        ax[2].legend()

        plt.show()

    def _plot_predictions(self, epoch):
        train_index = np.random.randint(0, len(self.X_train))
        val_index = np.random.randint(0, len(self.X_val))
        train_image, train_mask = self.X_train[train_index], self.y_train[train_index]
        val_image, val_mask = self.X_val[val_index], self.y_val[val_index]

        train_pred = self.model.predict(np.expand_dims(train_image, axis=0))[0]
        val_pred = self.model.predict(np.expand_dims(val_image, axis=0))[0]

        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        ax[0, 0].imshow(train_image)
        ax[0, 0].set_title('Train Image')
        ax[0, 1].imshow(train_mask.squeeze(), cmap='gray')
        ax[0, 1].set_title('Train Mask')
        ax[0, 2].imshow(train_pred.squeeze(), cmap='gray')
        ax[0, 2].set_title('Train Prediction')

        ax[1, 0].imshow(val_image)
        ax[1, 0].set_title('Val Image')
        ax[1, 1].imshow(val_mask.squeeze(), cmap='gray')
        ax[1, 1].set_title('Val Mask')
        ax[1, 2].imshow(val_pred.squeeze(), cmap='gray')
        ax[1, 2].set_title('Val Prediction')
        plt.show()

# Create the model
print("Creating the model...")
model = m_unet(
    input_size=input_size,
    learning_rate=learning_rate,
    dropout_rate=dropout_rate
)

# Compile the model with binary_crossentropy loss, Precision, Recall, and Dice Coefficient
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), dice_coefficient]
)

# Define callbacks
checkpoint = ModelCheckpoint("I:/MJU_project/Step1/data/models/model_suseo.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min")
early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="min")
visualize_callback = VisualizeCallback(X_train, y_train, X_val, y_val, interval=10)

# Train the model
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping, visualize_callback],
    verbose=1
)

print("Training completed. Best model saved as 'model_suseo.h5'.")