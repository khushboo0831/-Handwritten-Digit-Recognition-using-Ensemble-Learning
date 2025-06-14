## 🧠 Handwritten Digit Recognition using Ensemble Learning

### 📌 Overview

This project develops a powerful digit recognition model by combining **Convolutional Neural Network (CNN)**, **Support Vector Machine (SVM)**, and **Random Forest (RF)** classifiers. The ensemble method enhances accuracy and generalization by leveraging the strengths of each algorithm through **majority voting**.

---

### 🎯 Problem Statement

Handwritten digit recognition is a classic image classification challenge with real-world applications such as:

* **Postal code scanning**
* **Bank cheque processing**
* **Automated form reading**

The goal is to accurately identify digits (0–9) from 28x28 grayscale images using a hybrid learning approach.

---

### 🧰 Tech Stack

* **Python**
* **TensorFlow/Keras** – for deep learning (CNN)
* **scikit-learn** – for SVM and Random Forest
* **Pandas, NumPy, Matplotlib** – for preprocessing and visualization
* **Jupyter Notebook** – for experimentation

---

### 📂 Dataset Used

**Kaggle Digit Recognizer Dataset** (based on MNIST):

* `train.csv`: 785 columns (1 label + 784 pixels)
* `test.csv`: 784 columns of pixel values
* Each image: **28x28 grayscale**

---

### ⚙️ Workflow

#### 🔄 Data Preprocessing

* Normalized pixel values (0–255 → 0–1)
* Split data into training and validation sets

#### 🏗️ Model Building

* **CNN**: Used Conv2D, MaxPooling, and Dense layers
* **SVM**: Trained on flattened pixel data
* **Random Forest**: Trained using default scikit-learn parameters

#### 🤝 Ensemble Learning Strategy

* Each model predicts on the validation set
* Final prediction is based on **majority voting**
* Evaluated accuracy of individual and combined models

---

### 📊 Results

| Model         | Accuracy     |
| ------------- | ------------ |
| CNN           | \~97%        |
| SVM           | \~96%        |
| Random Forest | \~95%        |
| **Ensemble**  | ✅ **97.95%** |

---

### ✅ Conclusion

The ensemble model effectively combines **deep learning** and **traditional machine learning** to deliver a robust and generalizable digit recognition system. The hybrid approach improves reliability in varied image recognition tasks.

---

### 💡 Future Improvements

* Implement **weighted voting** or **stacking** ensemble techniques
* Apply **data augmentation** to improve CNN performance
* Use **GridSearchCV** for advanced hyperparameter tuning

---

### 👩‍💻 Owner

**Khushboo Verma**

This project reflects the power of ensemble learning by integrating multiple classification models to tackle real-world image classification problems effectively.

