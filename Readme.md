
Title: HematoVision: Blood Cell Classification Using Transfer Learning
---

1. Project Overview

Objective:
To build an efficient deep learning model using transfer learning to classify blood cells (Lymphocyte, Monocyte, Neutrophil, Eosinophil) and integrate it into a Flask-based web application for easy usage in healthcare diagnostics and education.

Tech Stack:

Language: Python

Frameworks: TensorFlow, Keras, Flask

Libraries: OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

Frontend: HTML (Jinja2), Bootstrap

Model: MobileNetV2 (Transfer Learning)

Data: Blood Cell Images (12,000+ images)

---

2. System Architecture

Flow:

1. User uploads an image via web UI.


2. Flask handles the request.


3. The image is preprocessed.


4. Trained model (Blood Cell.h5) predicts the cell class.


5. Prediction is returned on a result page.


3. Dataset Details

Source: Kaggle Blood Cell Dataset

Format: JPEG images categorized into folders by class:

Eosinophil

Lymphocyte

Monocyte

Neutrophil


Size: ~12,500 images

---

4. Preprocessing & Augmentation

Image resizing (typically 224x224)

Normalization

Augmentation methods (flip, rotation) applied during training

Used ImageDataGenerator for training-validation split

---

5. Model Building – MobileNetV2

Pre-trained base: MobileNetV2

Layers:

MobileNetV2 (without top layer)

Flatten

Dropout

Dense(4) with Softmax


Loss: Categorical Crossentropy

Optimizer: Adam

Training: 5 epochs

Saved model: Blood Cell.h5


6. Evaluation Metrics

Accuracy: ~90%

Tools Used: confusion matrix, classification report (from sklearn.metrics)

Validation: Image test samples used for model predictions

7. Web App Development

Framework: Flask

Pages:

home.html – upload form

result.html – display predictions


Functionality:

Upload → Prediction → Result Display


Commands:

Run: python app.py

Access: http://127.0.0.1:50

8. Use Cases

Automated diagnostics in pathology labs

Remote consultation (Telemedicine)

Training tool for medical students


9. Tools & Prerequisites

Anaconda Navigator

Libraries to install:

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow flask opencv-python


