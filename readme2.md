# Age Recognition Using EfficientNetB0

This project is focused on developing a deep learning model to classify human age groups using images. The dataset contains images categorized into four age ranges: 6-20, 25-30, 42-48, and 60-98 years. The model leverages transfer learning with the pre-trained EfficientNetB0 architecture to achieve robust performance on the task of age recognition.

## Dataset

The dataset used for this project can be found on Kaggle: [Age Recognition Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/age-recognition-dataset).

### Age Groups
The images are classified into the following four age groups:
- 6-20 years
- 25-30 years
- 42-48 years
- 60-98 years

### Directory Structure
The dataset is organized into folders corresponding to each age group:
```
Dataset/
    ├── 6-20/
    ├── 25-30/
    ├── 42-48/
    ├── 60-98/
```

## Project Overview

The project consists of the following steps:

1. **Data Loading and Preprocessing**:
   - The image file paths are collected from the dataset folders.
   - Labels are assigned based on the folder names.
   - The dataset is split into training and test sets using stratified sampling to maintain the distribution of classes.
   - Image augmentation is applied to the training set to improve model generalization.

2. **Model Architecture**:
   - A pre-trained `EfficientNetB0` model is used as the base model. The top layers are removed and replaced with custom layers, including a `GlobalAveragePooling2D`, `Dropout`, and `Dense` layers.
   - The last 20 layers of the `EfficientNetB0` base model are fine-tuned to adapt to the age classification task.
   - The output layer consists of four nodes corresponding to the four age groups, with a `softmax` activation for multi-class classification.

3. **Training the Model**:
   - The model is compiled using the Adam optimizer with a learning rate of `0.01` and categorical cross-entropy loss.
   - Training is performed with early stopping to prevent overfitting, with patience set to 5 epochs based on validation loss.
   
4. **Prediction**:
   - A sample image is loaded and preprocessed.
   - The model predicts the age group of the image, and the predicted label is displayed.

## Code Summary

### 1. Data Preprocessing
- Image paths and labels are collected from the dataset.
- A stratified split ensures proportional class distribution in training and test sets.
- Data augmentation (rotation, shifts, shear, zoom, horizontal flip) is applied to the training images.

### 2. Model Building
- The pre-trained EfficientNetB0 model is used as the base.
- Custom layers are added for pooling, dropout (to reduce overfitting), and a dense layer for classification.

### 3. Model Training
- The model is trained with `ImageDataGenerator` to load images from the dataset and apply augmentation.
- Early stopping is implemented to stop training when the validation loss stops improving.

### 4. Prediction
- The model predicts the class label for a given image, and the result is displayed.

## Results
- The model predicts the age group of an image based on visual features. The prediction is displayed alongside the input image for validation.

## Instructions to Run

1. Clone the repository and download the dataset from Kaggle.
2. Set up the environment with the required dependencies (`TensorFlow`, `Keras`, `Pandas`, `Matplotlib`, `PIL`, `OpenCV`).
3. Run the Jupyter notebook or script to train the model.
4. Use the trained model to predict age group labels for new images.

## Acknowledgments

- The dataset used is provided by [Rashik Rahman Pritom](https://www.kaggle.com/rashikrahmanpritom) on Kaggle.
