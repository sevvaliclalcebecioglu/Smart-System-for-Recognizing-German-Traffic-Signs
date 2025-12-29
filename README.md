# 🚦 Intelligent Traffic Sign Recognition System (Real-Time Image Prediction)

This project presents a **deep learning–based intelligent system** capable of recognizing **German traffic signs** from images using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset from Kaggle.

The developed system allows users to upload an image of a traffic sign and instantly predicts the corresponding sign class with high accuracy.

---

## 🎯 Project Objective

- Build a **Convolutional Neural Network (CNN)** to classify German traffic signs  
- Achieve high classification accuracy on a multi-class image dataset  
- Enable **real-time traffic sign prediction** from uploaded images  
- Demonstrate an end-to-end computer vision pipeline  

---

## 📂 Dataset Description

- **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)  
- **Source:** Kaggle  
- **Number of Classes:** 43 traffic sign categories  
- **Total Images:** 39,209 training images  

Each image represents a specific traffic sign captured under varying lighting, angles, and environmental conditions.

---

## 🔍 Data Preparation & Preprocessing

- Loaded image paths and labels from CSV files (`Train.csv`, `Test.csv`, `Meta.csv`)
- Read images using OpenCV
- Resized all images to **32×32 pixels** for efficient CNN processing
- Normalized pixel values to the range `[0, 1]`
- Applied **one-hot encoding** to class labels
- Split data into:
  - **80% Training**
  - **20% Testing**

---

## 🧠 Model Architecture

A **Convolutional Neural Network (CNN)** was designed using Keras:

- Convolutional Layer (32 filters, 3×3) + ReLU  
- Max Pooling Layer  
- Convolutional Layer (64 filters, 3×3) + ReLU  
- Max Pooling Layer  
- Flatten Layer  
- Fully Connected Dense Layer (128 units)  
- Dropout (0.5) to prevent overfitting  
- Output Layer with **Softmax activation**  

---

## ⚙️ Model Training

- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Metric:** Accuracy  
- **Epochs:** 15  
- **Training Environment:** GPU/CPU compatible  

---

## 📊 Model Performance

### Training & Validation Results
- Validation accuracy rapidly exceeded **99%**
- Stable convergence with minimal overfitting

### Final Test Performance
- **Test Accuracy:** **99.54%**

This demonstrates the model’s strong generalization capability on unseen data.

---

## 📈 Performance Visualization

- Accuracy vs Epoch (Training & Validation)
- Loss vs Epoch (Training & Validation)

These visualizations confirm consistent learning and effective optimization.

---

## 💾 Model Deployment

The trained CNN model was saved in `.keras` format and can be directly used for:
- Real-time image inference
- Web or mobile applications
- Intelligent transportation systems

---

## 🚗 Real-World Applications

This system can be applied in:

- Advanced Driver Assistance Systems (ADAS)  
- Autonomous vehicles  
- Smart traffic monitoring systems  
- Driver education platforms  
- Computer vision–based safety applications  

---

## 🛠️ Technologies Used

- Python  
- OpenCV  
- NumPy, Pandas  
- Matplotlib  
- Scikit-learn  
- TensorFlow / Keras  
- KaggleHub  

---

## 🧾 Conclusion

- A high-accuracy CNN model was successfully developed for traffic sign recognition.
- The model achieved **near-perfect classification accuracy** on the test dataset.
- The project demonstrates the effectiveness of deep learning in **computer vision and intelligent transportation systems**.
- Users can upload images and receive **instant traffic sign predictions**.

---
