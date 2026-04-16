import json
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

def feature_extraction(df):
    # Extract features from the DataFrame
    features = pd.DataFrame({
        'mean_torque': [df['Torque (Nm)'].mean()],
        'max_torque': [df['Torque (Nm)'].max()],
        'min_torque': [df['Torque (Nm)'].min()],
        'std_torque': [df['Torque (Nm)'].std()],
        'mean_current': [df['Current (V)'].mean()],
        'max_current': [df['Current (V)'].max()],
        'min_current': [df['Current (V)'].min()],
        'std_current': [df['Current (V)'].std()],
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

def visualize_features(all_features, labels):
    """
    Visualize the distribution of each feature across different classes using box plots.
    """
    class_names = ['N', 'NS', 'OT', 'P', 'UT']
    features_df = all_features.copy()
    features_df['label'] = labels
    
    num_features = len(all_features.columns)
    cols = 4  # Number of columns in subplot grid
    rows = (num_features + cols - 1) // cols  # Calculate rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(all_features.columns):
        ax = axes[i]
        data = [features_df[features_df['label'] == j][feature] for j in range(5)]
        ax.boxplot(data, tick_labels=class_names)
        ax.set_title(f'{feature}')
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show(block=False)


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

    new_features = False # Set to True to extract features from CSV files, False to load from JSON
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "features_and_labels.json")):
        new_features = True

    if new_features == True:
        base_path = os.path.join(os.path.dirname(__file__), "..", "Data fra tidligere project", "Dataset", "Intrinsic data")
        # Load features for each label group
        featuresN = feature_build(os.path.join(base_path, "N"))
        featuresNS = feature_build(os.path.join(base_path, "NS"))
        featuresOT = feature_build(os.path.join(base_path, "OT"))
        featuresP = feature_build(os.path.join(base_path, "P"))
        featuresUT = feature_build(os.path.join(base_path, "UT"))

        # Concatenate all features into one DataFrame
        all_features = pd.concat([featuresN, featuresNS, featuresOT, featuresP, featuresUT], ignore_index=True)

        # Create labels: 0=N, 1=NS, 2=OT, 3=P, 4=UT
        labels = ([0] * len(featuresN) + 
                [1] * len(featuresNS) + 
                [2] * len(featuresOT) + 
                [3] * len(featuresP) + 
                [4] * len(featuresUT))
        labels = np.array(labels)

        save_path = os.path.join(os.path.dirname(__file__), "features_and_labels.json")
        with open(save_path, 'w') as f:
            json.dump({
                'features': all_features.to_dict(orient='list'),
                'labels': labels.tolist()
            }, f)

    else:
        # Load features and labels from JSON file
        save_path = os.path.join(os.path.dirname(__file__), "features_and_labels.json")
        with open(save_path, 'r') as f:
            data = json.load(f)
        all_features = pd.DataFrame(data['features'])
        labels = np.array(data['labels'])

    # Visualize features
    #visualize_features(all_features, labels)

    print(f"labels: {labels}")
    print(f"Total samples: {len(all_features)}")
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

    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    elapsed_time = time.time() - start_time
    print(f"\nExecution time: {elapsed_time:.2f} seconds")


    # Display confusion matrix
    display_confusion_matrix(cm, 'SVM Confusion Matrix')

    




if __name__ == "__main__":
    main()

