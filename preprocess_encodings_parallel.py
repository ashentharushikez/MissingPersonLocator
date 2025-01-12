import face_recognition
import os
import cv2
import numpy as np
from pathlib import Path
import pickle
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns  # For better visuals

# Step 1: Load Images and Create Labels
# Path to dataset with subfolders for each person
dataset_path = Path(r"D:\Computer science\Research\FacialRecognizor3\kaggle_faces")

# Lists to store face encodings and corresponding labels
encodings = []
labels = []

# Loop through folders and images to extract face data
for person_folder in dataset_path.iterdir():
    if person_folder.is_dir():  # Check if it’s a folder
        person_name = person_folder.name  # Folder name is the label
        for image_path in person_folder.glob("*.jpg"):  # Get all .jpg files
            image = face_recognition.load_image_file(image_path)  # Load the image
            face_enc = face_recognition.face_encodings(image)  # Get face encoding
            if face_enc:  # If a face is found
                encodings.append(face_enc[0])  # Save the encoding
                labels.append(person_name)  # Save the label

# Step 2: Split data into training and testing sets
# 70% for training, 30% for testing
X_train, X_test, y_train, y_test = train_test_split(encodings, labels, test_size=0.3, random_state=42)

# Step 3: Train Classifier on Training Data
# SVM (Support Vector Machine) classifier
clf = svm.SVC(gamma='scale', probability=True)

# Lists to track accuracy (for visualization)
train_accuracies = []
test_accuracies = []

# Train the model with training data
clf.fit(X_train, y_train)

# Measure accuracy after training
train_acc = accuracy_score(y_train, clf.predict(X_train))  # Training accuracy
test_acc = accuracy_score(y_test, clf.predict(X_test))  # Testing accuracy

# Save accuracies for visualization
train_accuracies.append(train_acc)
test_accuracies.append(test_acc)

# Step 4: Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Training Accuracy')  # Plot training accuracy
plt.plot(test_accuracies, label='Test Accuracy')  # Plot testing accuracy
plt.xlabel('Iterations')  # X-axis label
plt.ylabel('Accuracy')  # Y-axis label
plt.title('Accuracy During Training (SVM)')  # Title of the graph
plt.legend(loc='best')  # Add legend
plt.grid(True)  # Add grid
plt.show()

# Step 5: Generate Confusion Matrix
y_pred = clf.predict(X_test)  # Predictions on test data
cm = confusion_matrix(y_test, y_pred)  # Confusion matrix

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(labels), yticklabels=set(labels))  # Plot heatmap
plt.xlabel("Predicted Labels")  # X-axis label
plt.ylabel("True Labels")  # Y-axis label
plt.title("Confusion Matrix")  # Title of the graph
plt.show()
