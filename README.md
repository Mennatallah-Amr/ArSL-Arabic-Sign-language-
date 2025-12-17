# Arabic Sign Language Recognition System

## Overview

This project presents an **Arabic Sign Language (ArSL) recognition system** based on deep learning and computer vision techniques. A **Convolutional Neural Network (CNN)** with a **MobileNetV2 backbone** is used to classify Arabic hand signs, while **MediaPipe** and **OpenCV** enable real-time hand detection and inference.

The system focuses on recognizing **31 Arabic alphabet characters** and is optimized for efficient execution using **GPU acceleration**.

---

## 1. Introduction

Sign language recognition is an essential component of assistive technologies aimed at reducing communication barriers for individuals with hearing impairments. Despite extensive research on sign language recognition, **Arabic Sign Language remains relatively underrepresented** in automated recognition systems.

This project addresses the problem of Arabic hand sign classification by combining **transfer learning** with **real-time hand tracking**. The system supports both **offline model training** and **real-time recognition**, making it suitable for educational, assistive, and research-oriented applications.

---

## 2. System Architecture

The system consists of two main components:

### 2.1 Model Training Module

* CNN based on **MobileNetV2**, pretrained on ImageNet
* Implemented using **PyTorch**
* Designed for multi-class classification of Arabic hand signs
* Transfer learning is used to reduce training time and improve generalization

### 2.2 Real-Time Recognition Module

* **MediaPipe** for hand detection and landmark extraction
* **OpenCV** for video capture and visualization
* Detected hand regions are preprocessed and passed to the trained CNN for inference

---

## 3. Model Design and Training

### Model Architecture

* **Backbone:** MobileNetV2 (ImageNet pretrained)
* **Classifier Head:**

  * Fully connected layers
  * Batch Normalization
  * Dropout for regularization

### Training Configuration

* **Loss Function:** CrossEntropyLoss
* **Optimizer:** AdamW

### Training Strategy

* Transfer learning with frozen feature extractor
* Data augmentation to enhance generalization
* Mixed precision training (AMP)
* Learning rate scheduling
* Early stopping and model checkpointing

---

## 4. Technologies Used

* Python
* PyTorch and Torchvision
* MediaPipe
* OpenCV
* NumPy
* Matplotlib and Seaborn
* CUDA for GPU acceleration

---

## 5. Project Structure

```
ArSL---Arabic-Sign-Language/
├── hand_tracking.py        # Real-time hand tracking and inference
├── arabic-sign-language   # Model training script
├── class_names.txt        # Arabic sign class labels
├── README.md
├── LICENSE
```

The dataset and trained model weights are not included in the repository due to size constraints.

---

## 6. Usage

### 6.1 Model Training

To train the CNN model on the dataset:

```bash
python arabic-sign-language
```

### 6.2 Real-Time Inference

To run real-time Arabic sign recognition using a webcam:

```bash
python hand_tracking.py
```

---

## 7. Evaluation

The trained model is evaluated using:

* Training and validation accuracy
* Classification report
* Normalized confusion matrix

These metrics provide insight into overall performance and class-wise behavior.

---

## 8. Applications

* Assistive communication systems for individuals with hearing impairments
* Educational tools for learning Arabic Sign Language
* Human–computer interaction systems
* Research in gesture recognition and computer vision

---

## 9. Contributors

This project was developed as a **group academic project** for an Artificial Intelligence course.

---

## 10. License

This project is licensed under the **MIT License**.
