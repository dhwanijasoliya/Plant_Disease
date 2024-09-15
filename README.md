# Plant_Disease

# **Plant Disease Classification with CNN**

### **Table of Contents**
1. [Overview](#overview)
2. [Project Goals](#project-goals)
3. [Features](#features)
4. [Technologies Used](#technologies-used)
5. [Dataset](#dataset)
6. [Model Architecture](#model-architecture)
7. [Installation and Setup](#installation-and-setup)
8. [Usage](#usage)
9. [Results](#results)
10. [Future Work](#future-work)
11. [Contributing](#contributing)
12. [License](#license)

---

## **Overview**
This project focuses on **Convolutional Neural Network (CNN)** for the classification of images of plants according to their disease type. The model will be designed using images from the **PlantVillage Dataset** so that the accuracy of diagnosis of diseases within the agricultural field is enhanced.

---
## **Project Objectives
The goal of this project is to develop a deep learning model for recognizing the disease in the plants from their leaf images. It categorizes the plant images into several categories, for example- healthy or diseased, which will be useful to farmers and agriculturalists with respect to taking any action over the situation.

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

- You can download the dataset here: [PlantVillage Dataset](https://www.tensorflow.org/datasets/catalog/plant_village)
- Dataset Directory Structure:
  ```
  dataset_directory/
      ├── Bell_Pepper/
      ├── Potato/
      ├── Tomato/
      └── .
Architecture: Most of the Con2D with ReLU activation followed by MaxPooling2D. Data augmentation contains random flips and rotation to enhance generalization. Fully connected Dense layers have been used for final classification with softmax activation.
The model was compiled with Adam optimizer and Sparse Categorical Crossentropy as loss.
 
---
Installation and Setup
1. Clone the repository:
```bash
git clone https://github.com/dhwanijasoliya/plant-disease.git
cd plant-disease
```
2. Install dependencies:
Make sure you have Python installed. Install the essential packages in Python using the following commands:
```bash
pip install tensorflow matplotlib numpy
```

### **3. Dataset Setup:**
- Download the **PlantVillage Dataset** and place it into a folder called `PlantVillage/` inside the project directory.
  
  Sample dataset structure:
  ```bash
  plant-disease-classification/
     ├── PlantVillage/
     └── .
  ```

### **4. Run the script:**
```bash
python train_model.py
```

---
## **Usage**

### **Training the Model:**
1. Loaded dataset using `load_and_split_dataset()` function,
2. Created the model using the `build_model()` function.
3. Trained the model on the training set by calling `model.fit()` function.
4. Called the `evaluate_model()` function to evaluate the model on the test dataset.

```bash
python train_model.py
```

### **Prediction on New Images:**
The model will predict plant diseases in new images that can be done during the execution of test set predictions visualized against the actual classes vs the predicted classes as implemented in the `predict()` function.

---
## **Results**
This model reaches a **test accuracy** of over **XX%** on the PlantVillage Dataset after training. The user is allowed to view accuracy and loss curves that were generated along the course of training, which may provide further analysis capabilities.

Sample Prediction Output:

| Actual Class       | Predicted Class    | Confidence |
|---------------------|--------------------|------------|
| Tomato Healthy      | Tomato Healthy     | 98.76%     |
| Apple Scab    | Apple Scab      | 96.24%     |

---

## **License**
This project is licensed under the MIT License - see the file [LICENSE](LICENSE) for details.
---
