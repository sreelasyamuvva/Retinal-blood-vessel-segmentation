import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score



# ----------- Paths & Configurations -----------
train_path_images = r'C:\Users\sreel\MyProjects\unet_retinal_segmentation\train\images'
train_path_masks = r'C:\Users\sreel\MyProjects\unet_retinal_segmentation\train\masks'

img_width = 256
img_height = 256
img_channels = 3

# ----------- Load Data -----------
train_img_files = sorted(os.listdir(train_path_images))
train_mask_files = sorted(os.listdir(train_path_masks))

X = np.zeros((len(train_img_files), img_height, img_width, img_channels), dtype=np.uint8)
y = np.zeros((len(train_mask_files), img_height, img_width, 1), dtype=np.float32)  # Changed from bool

for i, file_name in enumerate(train_img_files):
    img = cv2.imread(os.path.join(train_path_images, file_name))
    img = cv2.resize(img, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X[i] = img

    mask = cv2.imread(os.path.join(train_path_masks, train_mask_files[i]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    
    # Ensure mask is binary
    mask = np.where(mask > 0.5, 1.0, 0.0)  # Convert to {0,1}
    y[i] = np.expand_dims(mask, axis=-1)

print(f"Sample mask values: {np.unique(y)}")  # Debugging check

def compute_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    dice = 2 * np.sum(y_true_flat * y_pred_flat) / (np.sum(y_true_flat) + np.sum(y_pred_flat))
    iou = np.sum(y_true_flat * y_pred_flat) / (np.sum(y_true_flat) + np.sum(y_pred_flat) - np.sum(y_true_flat * y_pred_flat))
    mcc = matthews_corrcoef(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, average='binary')
    recall = recall_score(y_true_flat, y_pred_flat, average='binary')
    f1 = f1_score(y_true_flat, y_pred_flat, average='binary')

    # Specificity Calculation
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))  # True Negatives
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))  # False Positives
    specificity = tn / (tn + fp)

    # AUC Score
    auc_score = roc_auc_score(y_true_flat, y_pred_flat)
    
    return dice, iou, mcc, precision, recall, f1, specificity, auc_score


# # ----------- Spatial Attention Layer -----------
# def spatial_attention(x):
#     avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
#     max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
    
#     avg_pool = tf.keras.layers.Dense(x.shape[-1] // 8, activation='relu')(avg_pool)  # Squeeze
#     max_pool = tf.keras.layers.Dense(x.shape[-1] // 8, activation='relu')(max_pool)

#     attention = tf.keras.layers.Add()([avg_pool, max_pool])
#     attention = tf.keras.layers.Dense(x.shape[-1], activation='sigmoid')(attention)  # Scale

#     attention = tf.keras.layers.Multiply()([x, attention])  # Apply Attention
#     return attention



# ----------- Define U-Net Model (with attention)-----------
def unet_model(input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x) 
        #x = tf.keras.layers.Dropout(0.3)(x) 

        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        #x = tf.keras.layers.BatchNormalization()(x)  
        #x = tf.keras.layers.Dropout(0.3)(x)  

        return x

    c1 = conv_block(inputs, 64)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2 = conv_block(p1, 128)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3 = conv_block(p2, 256)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

    # Bottleneck
    c4 = conv_block(p3, 512)

    # Apply Spatial Attention After Bottleneck
    #sa = spatial_attention(c4)  # 👈 Inserting SA Between Encoder & Decoder

    # Decoder with Conv2DTranspose (Upsampling)
    def up_block(x, skip, filters):
        x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=(2,2), padding='same')(x)
        x = tf.keras.layers.concatenate([x, skip])
        x = conv_block(x, filters)
        return x

    u1 = up_block(c4, c3, 256)
    u2 = up_block(u1, c2, 128)
    u3 = up_block(u2, c1, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(u3)

    return tf.keras.Model(inputs, outputs)

# ----------- Build & Compile Model -----------
model = unet_model((img_width, img_height, img_channels))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ----------- Training Callbacks -----------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('DR_UNetmodel.keras', save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

# ----------- Train Model -----------
results = model.fit(X, y, validation_split=0.1, batch_size=4, epochs=25, callbacks=callbacks)



test_image_path = r"C:\Users\sreel\MyProjects\unet_retinal_segmentation\test\images\10.png"
test_mask_path = r"C:\Users\sreel\MyProjects\unet_retinal_segmentation\test\masks\10.png"  # Updated mask path!


# Load test image
test_img = cv2.imread(test_image_path)
test_img = cv2.resize(test_img, (img_width, img_height))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
test_img = np.expand_dims(test_img, axis=0)  # Add batch dimension


# Load test mask
true_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
true_mask = cv2.resize(true_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
true_mask = np.expand_dims(true_mask, axis=-1)  # Ensure correct shape
true_mask = np.where(true_mask > 0.5, 1.0, 0.0)  # Convert to binary {0,1}



# Predict on test image
pred_mask = model.predict(test_img)[0]

# ----------- Threshold Prediction -----------
threshold = 0.11  # Lower threshold to capture faint vessels
pred_mask = (pred_mask > threshold).astype(np.uint8)

# ----------- Compute Evaluation Metrics -----------
dice, iou, mcc, precision, recall, f1, specificity, auc_score = compute_metrics(true_mask, pred_mask)

# Print evaluation results
print(f"Dice Coefficient: {dice:.4f}")
print(f"IoU: {iou:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# ----------- Visualize Results -----------
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)

#Visualing for 7th test image
plt.imshow(test_img[0])  # Remove batch dimension for visualization
plt.title("Test Image")

plt.subplot(1, 3, 2)
plt.imshow(true_mask[:, :, 0], cmap='gray')
plt.title("Ground Truth Mask")

plt.subplot(1, 3, 3)
plt.imshow(pred_mask[:, :, 0], cmap='gray')
plt.title("Predicted Mask")

plt.tight_layout()
plt.show()