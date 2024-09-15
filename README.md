# Plant_Disease

# **Plant Disease Classification with CNN**

### **Table of Contents**
1. [Overview](#overview)
2. [Project Goals](#project-goals)
3. [Features](#features)
4. [Technologies Used](#technologies-used)
5. [Dataset](#dataset)
6. [Usage](#usage)
7. [Results](#results)
8. [License](#license)

---

## **Overview**
The project involves classifying the different disease conditions in plants using a Convolutional Neural Network. This algorithm has been built using TensorFlow and Keras, which in themselves form important ingredients of the process due to its architecture through the use of several layers such as convolutional, pooling, and dense layers, among other techniques like data augmentation and preprocessing.

---
## Project Objectives
The project is aimed at the training of a CNN model in the classification of images of plants into their respective categories with an accuracy level as high as possible. This model could then be used in real-world applications to detect and diagnose diseases in plants automatically.

--- 

## **Features**
- **Data Preprocessing**: Resize, rescale, and augment images for better model generalization.
- **CNN Model**: A custom-built CNN architecture with multiple convolutional layers for learning image features.
- **Model Training and Evaluation**: It trains the model on the **PlantVillage Dataset** and visualizes and estimates the performance of the model on previously unseen data.
- **Prediction and Visualization**: Displays predictions on new images with a very high level of confidence.

---
## **Technologies Used**
- **TensorFlow**: Deep Learning framework for model construction and training of the CNN.
- **Keras**: The high-level API for building neural networks.
- **Matplotlib**: Plotting accuracy/loss curves and visualization.
- **NumPy**: Numerical operations.

## **Dataset**
 The dataset employed for this experiment is the **PlantVillage Dataset**, which contains thousands of images of healthy and diseased plants.
- Dataset Directory Structure:
  ```
  dataset_directory/
      ├── Bell_Pepper/
      ├── Potato/
      ├── Tomato/
      └── .
Architecture: Most of the Con2D with ReLU activation followed by MaxPooling2D. Data augmentation contains random flips and rotation to enhance generalization. Fully connected Dense layers have been used for final classification with softmax activation.
The model was compiled with Adam optimizer and Sparse Categorical Crossentropy as loss.


## **Usage**

### **Training the Model:**
1. Loaded dataset using `load_and_split_dataset()` function,
2. Created the model using the `build_model()` function.
3. Trained the model on the training set by calling `model.fit()` function.
4. Called the `evaluate_model()` function to evaluate the model on the test dataset.

### **Prediction on New Images:**
The model will predict plant diseases in new images that can be done during the execution of test set predictions visualized against the actual classes vs the predicted classes as implemented in the `predict()` function.

---
## **Results**
This model reaches a **test accuracy** of over **XX%** on the PlantVillage Dataset after training. The user is allowed to view accuracy and loss curves that were generated along the course of training, which may provide further analysis capabilities.

Sample Prediction Output:

| Actual Class       | Predicted Class    | Confidence |
|---------------------|--------------------|------------|
| Tomato Healthy      | Tomato Healthy     | 98.76%     |

---

## **License**
This project is licensed under the MIT License.
