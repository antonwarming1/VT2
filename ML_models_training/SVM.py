import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def load_csv(filepath):
    return pd.read_csv(filepath)

def feature_extraction(df):
    # Extract features from the DataFrame
    features = pd.DataFrame({
        'mean_torque': [df['Torque (Nm)'].mean()],
        'max_torque': [df['Torque (Nm)'].max()],
        'min_torque': [df['Torque (Nm)'].min()],
        'mean_current': [df['Current (V)'].mean()],
        'max_current': [df['Current (V)'].max()],
        'min_current': [df['Current (V)'].min()],
        #'mean_angle': [df['Angle (deg)'].mean()],
        #'max_angle': [df['Angle (deg)'].max()],
        #'min_angle': [df['Angle (deg)'].min()],
        'torque_slope_max': [df['Torque (Nm)'].diff().max()],
        'torque_slope_std': [df['Torque (Nm)'].diff().std()],
        'current_slope_max': [df['Current (V)'].diff().max()],
        'current_slope_std': [df['Current (V)'].diff().std()]
    })
    return features

def feature_build(path):
    features = pd.DataFrame()

    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            filepath = os.path.join(path, filename)
            df = load_csv(filepath)
            features = pd.concat([features, feature_extraction(df)], ignore_index=True)

    return features

            
    
    
    

def main():
    base_path = os.path.join(os.path.dirname(__file__), "..", "Data fra tidligere project", "Dataset", "Intrinsic data")
    pathN = os.path.join(base_path, "N")
    pathNS = os.path.join(base_path, "NS")
    pathOT = os.path.join(base_path, "OT")
    pathP = os.path.join(base_path, "P")
    pathUT = os.path.join(base_path, "UT")
    # Load features for each label group
    featuresN = feature_build(pathN)
    featuresNS = feature_build(pathNS)
    featuresOT = feature_build(pathOT)
    featuresP = feature_build(pathP)
    featuresUT = feature_build(pathUT)

    # Concatenate all features into one DataFrame
    all_features = pd.concat([featuresN, featuresNS, featuresOT, featuresP, featuresUT], ignore_index=True)

    # Create labels: 0=N, 1=NS, 2=OT, 3=P, 4=UT
    labels = ([0] * len(featuresN) + 
              [1] * len(featuresNS) + 
              [2] * len(featuresOT) + 
              [3] * len(featuresP) + 
              [4] * len(featuresUT))
    labels = np.array(labels)

    print(f"Total samples: {len(all_features)}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Split into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.2, random_state=42, stratify=labels)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM classifier
    clf = svm.SVC(kernel='rbf', random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test_scaled)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['N', 'NS', 'OT', 'P', 'UT']))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Display confusion matrix
    labels = ['N', 'NS', 'OT', 'P', 'UT']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title('SVM Confusion Matrix')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()

