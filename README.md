# Retinal-blood-vessel-segmentation
Retinal vessel segmentation using a U-Net model with Batch Normalization after the first conv block. Trained on DRIVE dataset to enhance diabetic retinopathy detection. Achieved strong results with Dice 78%, IoU 64%, F1 78%, showing effective vessel extraction.


Input images and masks are resized to **256x256 pixels**, and masks are binarized to `{0,1}`.

## 🧪 Evaluation

The script computes the following metrics from predicted and ground truth masks:

- Dice Coefficient
- Intersection over Union (IoU)
- Matthews Correlation Coefficient (MCC)
- Precision, Recall, F1-Score
- Specificity
- AUC Score

## 📸 Visualization

Visual output includes side-by-side comparisons of:
- Original Test Image
- Ground Truth Mask
- Predicted Mask

## 🚀 Getting Started

1. **Install Dependencies**

pip install tensorflow opencv-python scikit-learn matplotlib numpy

2. **Prepare Dataset**

Ensure your dataset directory is organized as:

Update the following paths in the script to match your setup:

train_path_images = r'path_to/train/images'
train_path_masks  = r'path_to/train/masks'
test_image_path   = r'path_to/test/images/sample.png'
test_mask_path    = r'path_to/test/masks/sample.png'

3. **Train the Model**
Run the training block which:
- Initializes the U-Net model with BatchNorm
- Uses binary crossentropy loss
- Employs ModelCheckpoint and EarlyStopping callbacks
Model weights will be saved as DR_UNetmodel.keras.

## 🧠 Using the Pretrained Model
To skip training and directly use the pretrained model that i attached (unet_model.keras) as inference
Load 'unet_model.keras' pretrained model to your script

4. **Evaluate & Visualize**
After training, the model will:
- Predict segmentation on a test image
- Apply thresholding (0.11) to the output
- Compute metrics like Dice, IoU, MCC, F1, etc.
- Display a side-by-side visualization of:
- Test Input
- Ground Truth Mask
- Predicted Mask

🛠️**Customization**
- Enable spatial attention by uncommenting the spatial_attention() function and inserting it after the bottleneck layer.
- Adjust dropout, batch size, learning rate, and threshold for dataset-specific tuning.
- Post-process output masks using morphological filters for edge refinement.
📌 License
This project is released under the MIT License.

Feel free to contribute, fork, or extend this work for other biomedical segmentation challenges!
