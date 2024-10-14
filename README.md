# Anomaly Detection in Fences Using Deep Learning

This project focuses on detecting anomalies in fencing systems using computer vision and deep learning techniques. The goal is to automate the detection of structural defects in fences, such as broken wires or deformations, by leveraging Convolutional Neural Networks (CNNs) with transfer learning. This project was developed during our Bachelor's graduation in Computer Systems.

## Project Overview

Fencing systems, especially in industrial and security-sensitive areas, require constant monitoring to detect structural issues that could compromise safety. Our project automates this process by utilizing deep learning models to analyze images of fences and identify anomalies in real-time.

### Key Features:
- **Convolutional Neural Networks (CNNs)**: We utilized pre-trained models such as ResNet and VGG for high-accuracy anomaly detection.
- **Transfer Learning**: By using transfer learning, we reduced training time while maintaining model performance.
- **Real-time Detection**: The system is capable of processing images continuously and flagging anomalies as soon as they are detected.
- **Environmental Robustness**: The model is designed to handle diverse environmental conditions, such as lighting and weather variations, with minimal false positives.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Google Colab
- ResNet, VGG (Transfer Learning)

## Dataset

The dataset includes images of normal and abnormal fences, captured under different conditions. It was manually labeled to help train the model to recognize specific types of anomalies.

## Model Architecture

We applied Convolutional Neural Networks (CNNs) with transfer learning, fine-tuning pre-trained models for our specific use case:
- **ResNet50**
- **VGG16**

These models were trained on a custom dataset using augmentation techniques to improve generalization.

## Project Workflow

1. **Data Collection & Preprocessing**: Images of fences were collected and labeled as either normal or containing anomalies.
2. **Model Training**: We used transfer learning to fine-tune pre-trained CNN models for anomaly detection.
3. **Real-time Detection**: The trained model was deployed in a pipeline capable of processing images and detecting anomalies in real-time.
4. **Evaluation & Optimization**: The model was evaluated for accuracy, precision, recall, and environmental robustness.

## Results

- **Precision**: 92%
- **Recall**: 89%
- **Accuracy**: 90%
- **False Positives**: Minimal false positives, even in varying environmental conditions.

## Usage

To run the project, you can access the Colab link below and follow the instructions for running the code.

[**Colab Link**](https://drive.google.com/file/d/1Zp7Tpt8Mow1myqfh_sYEsTJnIQnmw1NG/view?usp=sharing) - Link to the notebook where you can experiment with the model and dataset.

## Project Report

For more technical details, you can refer to the full project report [**[anomaly detection.pdf](https://github.com/user-attachments/files/17362557/anomaly.detection.pdf)
here**](#) (in French).

## How to Contribute

Feel free to clone this repository, raise issues, or suggest improvements! Contributions to improve accuracy or handle more complex anomalies are welcome.

