"""
SVM classification using all features from feature engineering. 
This script loads the selected features and labels, trains an SVM classifier, 
evaluates its performance, and visualizes the confusion matrix.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def load_csv(filepath):
    print(f"Loading CSV file: {filepath}")
    return pd.read_csv(filepath)


def display_confusion_matrix(cm, title='Confusion Matrix'):
    """
    Display the confusion matrix using a heatmap.
    """
    labels = ['N', 'NS', 'OT', 'P', 'UT']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def main():
    
    start_time = time.time()

    # Load features and labels from JSON file
    feature_path = os.path.join(os.path.dirname(__file__),"..", "Feature_engineering", "features_selected.csv")
    all_features = load_csv(feature_path)
    
    label_path = os.path.join(os.path.dirname(__file__),"..", "Feature_engineering", "labels.csv")
    labels_df = load_csv(label_path)
    labels = np.array(labels_df['label'])

    print(f"Total samples: {len(all_features)}")
    print(f"Number of features: {all_features.shape[1]}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Split into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42, stratify=labels)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM classifier
    clf = svm.SVC(kernel='rbf', random_state=42, decision_function_shape='ovr')
    clf.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test_scaled)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['N', 'NS', 'OT', 'P', 'UT']))

    # Print confusion matrix to terminal
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    elapsed_time = time.time() - start_time
    print(f"\nExecution time: {elapsed_time:.4f} seconds")


    # Display confusion matrix as heatmap
    display_confusion_matrix(cm, 'SVM Confusion Matrix')

    




if __name__ == "__main__":
    main()

