# Disease-Prediction
This project implements a complete machine learning pipeline using a Random Forest classifier to analyze structured tabular data provided in CSV format. The Python implementation handles data preprocessing, model training, performance evaluation, and visualization using popular Python libraries such as pandas, scikit-learn, matplotlib, and seaborn.

✨ Key Features
1.📥 Flexible CSV Input: Load and process custom training and testing datasets.

2.🧹 Data Cleaning: Automatically aligns and filters columns to ensure compatibility.

3.🧠 Model Training: Uses RandomForestClassifier from scikit-learn with customizable parameters.

4.📈 Evaluation Metrics:

   a.Accuracy score

   b.Confusion matrix (visualized with seaborn)

5.📊 Feature Importance Visualization: Highlights the top 10 most important features based on Gini impurity.

6.📂 Modular Design: Code is organized into clear functions, ready for extension or reuse in larger ML workflows.

## 📁 Project Structure

Disease-Prediction/
├── data/
│ ├── Training.csv
│ └── Testing.csv
├── main.py
├── requirements.txt
└── README.md

## 🚀 How to Run

1. Install dependencies:
pip install -r requirements.txt

2.Place your Training.csv and Testing.csv in the data/ directory.

3.Run the script: python main.py

## 📊 Output
confusion_matrix.png: Heatmap of classification results.

feature_importance.png: Bar chart of top 10 features.

## 🛠️ Dependencies

1.pandas

2.numpy

3.scikit-learn

4.matplotlib

5.seaborn
