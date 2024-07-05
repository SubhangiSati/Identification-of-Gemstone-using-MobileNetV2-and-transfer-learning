# Gemstone Classification using Transfer Learning with MobileNetV2

## Overview

This code implements a gemstone classification model using transfer learning with the MobileNetV2 architecture. The model is trained on a dataset of gemstone images, and its performance is evaluated using various metrics, including accuracy, loss, and a confusion matrix.

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

Ensure you have the required dependencies installed using:

```bash
pip install tensorflow matplotlib seaborn scikit-learn
```

## Usage

1. Download the gemstone dataset and organize it into train and test directories.

2. Update the `train_data_path` and `test_data_path` variables with the correct paths to your train and test datasets.

3. Optionally, set the `image_to_predict_path` variable to the path of a specific image for prediction.

4. Run the script to train the model, evaluate its performance, and make predictions.

```bash
python gemstone_classification.py
```

## Code Structure

- **Data Preprocessing:**
  - Images are loaded and preprocessed using data augmentation for training (`ImageDataGenerator`).
  
- **Model Architecture:**
  - MobileNetV2 is used as the base model with additional custom top layers for classification.
  - The model is compiled with categorical cross-entropy loss and Adam optimizer.

- **Training:**
  - The model is trained using the specified learning rate schedule and early stopping.

- **Evaluation:**
  - The model is evaluated on the test set, and accuracy and loss curves are plotted.

- **Prediction:**
  - A sample image is loaded, preprocessed, and the model makes predictions.

- **Confusion Matrix:**
  - Confusion matrix and classification report are generated for evaluating model performance.

## Hyperparameters

- `input_shape`: Input image dimensions (224x224x3).
- `num_classes`: Number of gemstone classes (87).
- `batch_size`: Batch size for training and testing (32).
- `epochs`: Number of training epochs (50).

## Customization

- Adjust the hyperparameters to suit your specific dataset and computing resources.
- Modify the learning rate schedule, early stopping parameters, or model architecture as needed.

## License

This code is licensed under the [MIT License](LICENSE).
