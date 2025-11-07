# Image Classification with a Convolutional Neural Network

This repository contains a full workflow for building a deep learning pipeline to classify and analyze annotated ultrasound liver images using a convolutional neural network (CNN). It covers dataset management, preprocessing, data augmentation, visualization, and implementation of the CNN for classification of liver health status.

I have a YouTube video where I walk through this notebook, found [HERE](https://www.youtube.com/watch?v=btbQuHFyA0I&t=852s). 

## Dataset

The original data can be found here: [https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset].

The data contains 735 images of liver ultrasound with labels: Normal, Benign, and Malignant. For each image there is a JSON representing polygon outlines of liver, mass, and outline areas.

## Main Steps

Please navigate to **liver_image_classification.ipynb** to see the whole flow with detailed steps.

- Environment Setup
    - Dependency installations and python modules
    - Dataset is downloaded from Kaggle via kagglehub
- Exploratory Data Analysis
    - Example images displayed with annotation overlayed to show outline, liver, mass areas for each class
    <img width="484" height="389" alt="image" src="https://github.com/user-attachments/assets/5b282b7a-899f-403c-9897-c33be5fe6a88" />
    - EDA includes data distribution, verifying consistency in format, and other EDA
- Data Transformation and Augmentation
    - Standardize image sizes
    - Augment images for equal sampling
    - Variations for better learning (e.g., flip, brightness, contrast, jitter adjustments)
    <img width="924" height="985" alt="image" src="https://github.com/user-attachments/assets/1a6df567-4604-4bae-9fc9-cfe547b51d93" />
- Prep for Modeling
    - Train/Test split
    - Manifest files for reproducibility
    - PyTorch dataset class
- CNN Model Implementation
    - Binary Classification CNN architecture
        - Four convolutional blocks, fully connected classifier, 6 input channels
    <img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/5aa2bd3b-a6bf-454a-871e-b19803c60f49" />
    <img width="1453" height="593" alt="image" src="https://github.com/user-attachments/assets/1347ab06-ae79-4314-a3a0-26742c722f4e" />
    - Multi-class Classification
- Training and Evaluation
    - Neural network model training loop with AdamW optimizer and BCEWithLogits loss

