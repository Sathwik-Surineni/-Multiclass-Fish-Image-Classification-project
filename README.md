# Multiclass Fish Image Classification

## Project Overview

This project implements a fish species classifier using transfer learning with MobileNetV2. It classifies images of different fish species by training a lightweight deep learning model and provides an easy-to-use web demo through Streamlit.

---

## Features

- Uses MobileNetV2 as a pretrained base model with a custom classification head
- Data augmentation and validation split built into the training pipeline
- Saves the best model (`best_model.h5`) based on validation accuracy
- Streamlit app (`app.py`) for interactive image upload and species prediction
- Evaluation script (`evaluate.py`) to generate classification reports and confusion matrix

---

## Folder Structure
```
project_folder/
│
├── data/ # Dataset folder containing subfolders of fish classes
│
├── best_model.h5 # Trained model file
├── class_indices.json # Class label mappings saved during training
├── train.py # Training script
├── evaluate.py # Evaluation script
├── app.py # Streamlit demo app
├── training_accuracy.png # Training accuracy plot (optional)
├── confusion_matrix.png # Confusion matrix plot (optional)
├── requirements.txt # Required Python packages
└── README.md # This file
```
### How to run (local)
1.Create and activate a virtual environment.

2.Install dependencies: pip install -r requirements.txt.

3.Unzip the dataset to a data/ folder, ensuring the structure is data/<class_name>/*.jpg.

4.Run the training script: python train.py. This will train the model for a few epochs and save best_model.h5.

5.Run the evaluation script: python evaluate.py. This will print a classification report and save a confusion matrix plot.

6.Launch the Streamlit app: streamlit run app.py. Open the local address in your browser, upload an image, and see the prediction.

### Deliverables
1.best_model.h5: The saved, trained model.

2.app.py: The Streamlit application for inference.

3.train.py: The script used for training the model.

4.evaluate.py: The script used to evaluate the model's performance.

5.requirements.txt: A list of all necessary Python libraries.

6.training_accuracy.png: A plot showing the training and validation accuracy.

7.confusion_matrix.png: A heatmap of the confusion matrix.

8.demo.mp4: A short explanatory video of the projec

### Requirements
tensorflow

streamlit

pillow

scikit-learn

matplotlib

seaborn

These are included in requirements.txt for easy installation.

### Notes
~This is a minimal working demo focusing on fast training and ease of use.

~Accuracy can be improved with more epochs, better augmentation, or more data.

~The Streamlit app is designed for quick testing of fish image classification.
