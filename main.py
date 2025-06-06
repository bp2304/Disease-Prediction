import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(train_path, test_path):
    train_data = pd.read_csv(train_path).iloc[:, :-1]
    test_data = pd.read_csv(test_path).iloc[:, :-1]

    common_cols = list(set(train_data.columns) & set(test_data.columns))
    train_data = train_data[common_cols]
    test_data = test_data[common_cols]

    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    x_combined = combined_data.iloc[:, :-1]
    y_combined = combined_data.iloc[:, -1]

    if y_combined.dtype == 'object' or y_combined.dtype.name == 'category':
        label_encoder = LabelEncoder()
        y_combined = label_encoder.fit_transform(y_combined)
    else:
        label_encoder = None

    n_train = train_data.shape[0]
    x_train, x_test = x_combined.iloc[:n_train], x_combined.iloc[n_train:]
    y_train, y_test = y_combined[:n_train], y_combined[n_train:]

    return x_train, x_test, y_train, y_test, label_encoder

def train_and_evaluate(x_train, x_test, y_train, y_test, label_encoder):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on test set: {accuracy:.4f}")

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    labels = label_encoder.classes_ if label_encoder else sorted(set(y_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    importances = model.feature_importances_
    feat_importance = pd.DataFrame({'Variables': x_train.columns, 'Importance': importances})
    feat_importance = feat_importance.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_importance, x='Importance', y='Variables', palette="Blues_d")
    plt.title("Top 10 Most Important Variables")
    plt.xlabel("Importance")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

def main():
    train_path = os.path.join("data", "Training.csv")
    test_path = os.path.join("data", "Testing.csv")
    x_train, x_test, y_train, y_test, le = load_and_preprocess(train_path, test_path)
    train_and_evaluate(x_train, x_test, y_train, y_test, le)

if __name__ == "__main__":
    main()
