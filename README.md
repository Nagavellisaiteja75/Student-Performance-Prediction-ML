# 🎓 Student Performance Prediction

> **Machine Learning Mini Project — Task 5**

A simple machine learning classification project that predicts whether a student is likely to **PASS or FAIL** based on academic and engagement-related features such as study hours, attendance, previous marks, assignment performance, and project requirements.

---

## 📌 Project Overview

Student performance can be influenced by several measurable factors. This project uses **Machine Learning** to identify patterns in student data and predict a student's final outcome as either **PASS** or **FAIL**.

The project follows a complete basic ML workflow:

**Data Collection → Data Cleaning → Feature Selection → Train-Test Split → Preprocessing → Model Training → Prediction → Evaluation**

For this project, **Logistic Regression** is used as the classification algorithm because the target variable contains two classes: `PASS` and `FAIL`.

---

## 🎯 Objectives

- Load and inspect a student performance dataset using Pandas.
- Clean and prepare the dataset for machine learning.
- Select relevant input features and the target variable.
- Split the data into training and testing sets.
- Preprocess numerical and categorical features.
- Train a Logistic Regression classification model.
- Evaluate the model using test accuracy and classification metrics.
- Test the trained model using three new student inputs.
- Display the predicted student outcomes.

---

## 📊 Dataset

The project uses a **synthetic Student Performance dataset created for educational purposes**.

It contains **120 student records** with both PASS and FAIL examples.

### Features

| Feature | Description |
|---|---|
| `Study_Hours` | Average hours spent studying |
| `Attendance` | Student attendance percentage |
| `Previous_Marks` | Marks obtained previously |
| `Assignment_Score` | Assignment performance score |
| `Project_Requirements` | Whether project requirements were met |
| `Result` | Target variable: PASS or FAIL |

### Target

**`Result`**

- `PASS`
- `FAIL`

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression is a supervised machine learning algorithm commonly used for classification problems.

In this project, it learns the relationship between the selected student features and the student's outcome.

The model workflow is:

```text
Student Features
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Logistic Regression
      ↓
PASS / FAIL Prediction
```

---

## ⚙️ Technologies Used

- **Python**
- **Pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **Scikit-learn** — preprocessing, model training and evaluation
- **Jupyter Notebook** — development and documentation
- **Matplotlib** — output visualization/screenshots

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

1. Check the dataset for missing values.
2. Fill missing numerical values using the median.
3. Fill missing categorical values using the mode.
4. Convert the categorical `Project_Requirements` feature into numerical form using one-hot encoding.
5. Split the dataset into:
   - **80% Training Data**
   - **20% Testing Data**
6. Standardize numerical features using `StandardScaler`.

The scaler is fitted on the training data and then applied to the test data to avoid data leakage.

---

## 📈 Model Evaluation

The Logistic Regression model was evaluated using the unseen test dataset.

### Result

**Test Accuracy: 83.33%**

The project also generates:

- Classification Report
- Confusion Matrix
- PASS/FAIL predictions
- Prediction confidence for new students

> **Note:** The dataset is synthetic and relatively small. Therefore, the reported accuracy is intended for educational demonstration and should not be interpreted as real-world academic prediction performance.

---

## 🧪 Testing With New Students

The trained model was tested using three new student profiles.

### Example 1

```text
Study Hours: 6
Attendance: 85%
Previous Marks: 72
Assignment Score: 82
Project Requirements: Met

Prediction: PASS
```

### Example 2

```text
Study Hours: 2.5
Attendance: 65%
Previous Marks: 42
Assignment Score: 45
Project Requirements: Not Met

Prediction: FAIL
```

### Example 3

```text
Study Hours: 8
Attendance: 94%
Previous Marks: 88
Assignment Score: 91
Project Requirements: Met

Prediction: PASS
```

These examples demonstrate how the trained model can be used to generate predictions for previously unseen student inputs.

---

## 📁 Project Structure

```text
student-performance-prediction-ml/
│
├── README.md
│
├── Task_5_Student_Performance_Prediction.ipynb
│
├── student_performance_pass_fail.csv
│
├── New_Student_Predictions.csv
│
├── Mini_Project_Report_200_300_Words.txt
│
└── screenshots/
    ├── dataset_preparation.png
    ├── model_accuracy.png
    └── new_student_predictions.png
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-prediction-ml.git
```

### 2. Open the project folder

```bash
cd student-performance-prediction-ml
```

### 3. Install required libraries

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
Task_5_Student_Performance_Prediction.ipynb
```

Run the notebook cells from top to bottom.

---

## 📷 Project Screenshots

### Dataset Preparation

![Dataset Preparation](screenshots/dataset_preparation.png)

### Model Accuracy

![Model Accuracy](screenshots/model_accuracy.png)

### New Student Predictions

![New Student Predictions](screenshots/new_student_predictions.png)

---

## ⚠️ Limitations

This project has several limitations:

- The dataset is synthetic and created for educational purposes.
- The dataset contains a relatively small number of records.
- Student performance depends on many factors that are not included.
- Model accuracy may change significantly with a larger real-world dataset.
- Predictions should not be treated as guaranteed academic outcomes.

---

## 🔮 Future Improvements

The project can be improved by:

- Using a larger real-world student dataset.
- Adding more relevant features such as study consistency, previous attendance history, exam difficulty, and learning behavior.
- Comparing multiple classification algorithms such as Decision Tree, Random Forest, KNN, and SVM.
- Using cross-validation for more reliable evaluation.
- Performing hyperparameter tuning.
- Building a web application where users can enter student details and receive a prediction.
- Deploying the trained model as an online ML application.

---

## 📚 Learning Outcomes

Through this project, the following concepts were practiced:

- Data cleaning
- Feature engineering
- Feature selection
- Categorical encoding
- Train-test splitting
- Feature scaling
- Supervised learning
- Binary classification
- Logistic Regression
- Model evaluation
- Making predictions on new data

---

## 👨‍💻 Author

**N. Sai Teja**

B.Tech — Computer Science & Engineering (Artificial Intelligence & Machine Learning)

Kommuri Pratap Reddy Institute of Technology

---

## 📄 Academic Project

**Course/Assignment:** Machine Learning Fundamentals  
**Task:** Task 5 — Mini Project  
**Project:** Student Performance Prediction

---

⭐ If you found this project useful, consider giving the repository a star!
