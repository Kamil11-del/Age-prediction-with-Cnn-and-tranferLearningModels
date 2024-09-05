# Age Detection Model Using EfficientNetB0

This project implements an age detection model using the **EfficientNetB0** architecture pre-trained on ImageNet. The model is fine-tuned to classify images of faces into three age groups: **young**, **middle-aged**, and **old**. The dataset used for this project can be found on Kaggle, and instructions for setting up the environment and running the code are provided below.

## Dataset

The dataset for this project can be downloaded from Kaggle at the following link: [Faces Age Detection Dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset).

## Installation

To run this project, you will need the following dependencies. You can install them using `pip` or `conda`.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your_username/age-detection-model.git
cd age-detection-model
```

### Step 2: Install the Required Libraries

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn tqdm
```

Or use `conda` if preferred:

```bash
conda create --name age-detection python=3.8
conda activate age-detection
conda install tensorflow keras numpy pandas matplotlib scikit-learn tqdm
```

### Step 3: Download the Dataset

1. Download the dataset from [here](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset).
2. Unzip the dataset and place the files in the appropriate directory in your project folder (e.g., `/input/faces-age-detection-dataset/`).

### Step 4: Run the Model

Once the dataset is set up, run the following command to train the model:

```bash
python train.py
```

## Model Architecture

The model is built using **EfficientNetB0** as the base model and is fine-tuned by training the last 20 layers. The architecture includes a **Global Average Pooling** layer, a **Dropout** layer for regularization, and a final **Dense** layer with a softmax activation to classify into three age categories.

## Usage

You can use the trained model to predict the age group of any image by providing the path to the image. An example function is included in the code to load an image, preprocess it, and make a prediction.

## Results

The model provides the predicted age group (young, middle, old) based on the facial image provided.

## License

This project is licensed under the MIT License.
```

Feel free to customize the repository URL or dataset path as necessary.
```
