# Assignment 9 — Image Classification using CNN (Cats vs Dogs)

## Objective
Build a Convolutional Neural Network (CNN) to automate classification of pet images into Cats and Dogs, for an animal welfare organization use case.

## Dataset
[Cats vs Dogs Classification Dataset — Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

*(Dataset is not uploaded to this repo per assignment instructions — download it directly from the Kaggle link above using the notebook's built-in Kaggle API download cell.)*

## Libraries Used
- TensorFlow / Keras — model building, training, `ImageDataGenerator` for preprocessing
- NumPy — array operations
- Matplotlib / Seaborn — visualizations (sample images, accuracy/loss curves, confusion matrix)
- scikit-learn — precision, recall, F1-score, confusion matrix
- Pillow (PIL) — image inspection

## Methodology
1. **Data Understanding** — explored folder structure, sample images, class counts, and image dimensions.
2. **Data Preprocessing** — resized all images to 128×128, normalized pixel values to 0–1, split 80/20 train/test using `ImageDataGenerator`.
3. **Model Development** — built a 3-block CNN (Conv2D + MaxPooling2D, filters 32 → 64 → 128) followed by Flatten and Dense layers, compiled with Adam optimizer and binary crossentropy loss, trained for 10 epochs.
4. **Model Evaluation** — measured test accuracy, precision, recall, F1-score; generated a confusion matrix and accuracy/loss vs epoch graphs.
5. **Conclusion** — summarized findings and CNN's advantages/limitations for this task.

## CNN Architecture
```
Input (128 x 128 x 3)
  -> Conv2D(32, 3x3, ReLU) -> MaxPooling2D(2x2)
  -> Conv2D(64, 3x3, ReLU) -> MaxPooling2D(2x2)
  -> Conv2D(128, 3x3, ReLU) -> MaxPooling2D(2x2)
  -> Flatten
  -> Dense(128, ReLU)
  -> Dense(1, Sigmoid)
```
Optimizer: Adam | Loss: Binary Crossentropy | Metric: Accuracy | Epochs: 10

## Results
*(Fill in after running the notebook in Google Colab)*
- Test Accuracy: **[XX]%**
- Precision: **[XX]**
- Recall: **[XX]**
- F1-Score: **[XX]**

## Conclusion
This project built a Convolutional Neural Network to classify pet images into Cats and Dogs, achieving a test accuracy of **[XX]%** after 10 training epochs. Convolution and pooling layers are central to why CNNs work well on images: convolution layers detect local patterns like edges and textures regardless of where they appear, while pooling layers reduce dimensionality and add robustness to small shifts in the image. This makes CNNs far more efficient at image tasks than a standard ANN, which loses spatial relationships between pixels when flattening the image immediately. One limitation is that this CNN can overfit when training data lacks variety in lighting, angle, or occlusion — techniques like data augmentation, dropout, or transfer learning could improve robustness in future iterations.
