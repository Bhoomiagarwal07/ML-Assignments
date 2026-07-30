# Assignment 8: Handwritten Digit Recognition using ANN

## Objective
Develop an Artificial Neural Network (ANN) to classify handwritten digits (0–9) from the MNIST dataset, simulating an automated postal-code digit recognition system for a postal service organization.

## Dataset Link
[MNIST in CSV — Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

The dataset is **not included in this repository**. Download `mnist_train.csv` and `mnist_test.csv` directly from the Kaggle link above and place them in the working directory (or Colab runtime) before running the notebook.

## Libraries Used
- `pandas` — data loading and exploration
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — visualization (sample digits, confusion matrix, accuracy/loss curves)
- `scikit-learn` — train/test split, confusion matrix, classification report
- `tensorflow` / `keras` — building and training the ANN

## Methodology
1. **Data Understanding** — Loaded the MNIST CSV data with Pandas, inspected the first five records, identified the 784 pixel columns as input features and the `label` column as the target variable, and visualized a sample digit.
2. **Data Preprocessing** — Checked for missing values, separated features (X) from the target (y), normalized pixel values from the 0–255 range to 0–1, split the data into 80% training / 20% testing, and one-hot encoded the labels using `to_categorical`.
3. **Model Development** — Built a Sequential ANN with two hidden layers and trained it for 10 epochs using the Adam optimizer and categorical crossentropy loss.
4. **Model Evaluation** — Evaluated test accuracy, generated a confusion matrix and classification report, and plotted accuracy/loss curves across epochs.
5. **Conclusion** — Summarized findings, the role of hidden layers, an advantage of Deep Learning over traditional ML, and a limitation of ANN.

## Model Architecture

| Layer | Type | Units | Activation |
|---|---|---|---|
| Input | — | 784 (28×28 flattened) | — |
| Hidden Layer 1 | Dense | 128 | ReLU |
| Hidden Layer 2 | Dense | 64 | ReLU |
| Output Layer | Dense | 10 | Softmax |

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metric:** Accuracy
- **Epochs:** 10

## Results

- **Test Accuracy:** 96.91%
- **Test Loss:** 0.1204
- Recall was slightly lower for digits 3, 5, and 8 (0.95 each) compared to digits 0 and 6 (0.99 each), per the classification report.
- Validation loss began rising slightly after epoch 6 while training loss kept falling — a sign of mild overfitting in later epochs.
- Confusion Matrix, full Classification Report, and Accuracy/Loss vs Epoch plots: see `Assignment-8.ipynb`

## Conclusion
This project implemented an Artificial Neural Network to classify handwritten digits from the MNIST dataset, achieving a test accuracy of 96.91% with a simple architecture of two hidden layers (128 and 64 neurons). The classification report confirmed that performance was fairly consistent across all ten digit classes, with slightly lower recall for digits 3, 5, and 8, which are visually similar to one another. Hidden layers are central to an ANN's ability to learn: each layer builds progressively more abstract representations of the input pixels, letting the network capture non-linear patterns like curves and strokes that a single-layer model could not. Compared to traditional Machine Learning models (e.g., Logistic Regression or Decision Trees) that rely on hand-engineered features, this Deep Learning model learned relevant features directly from raw pixel data, reaching over 96% accuracy without any manual feature design. However, a clear limitation showed up in training: validation loss began rising slightly after epoch 6 even as training loss kept falling, indicating mild overfitting. A plain ANN also treats each pixel independently and ignores the image's 2D spatial structure, which is why Convolutional Neural Networks (CNNs) typically outperform ANNs on image classification tasks by explicitly modeling spatial relationships.

## Repository Structure
```
├── Assignment-8.ipynb   # Full notebook: data understanding, preprocessing, model, evaluation
└── README.md             # This file
```
