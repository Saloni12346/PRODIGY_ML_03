import os
import zipfile
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Set paths to the dataset
data_dir = "/content/dogvscats.zip"

# Image parameters
IMG_SIZE = 64  # Resize all images to 64x64

# Function to load and preprocess data
def load_data(data_dir):
    images = []
    labels = []

    # Open the zip file
    with zipfile.ZipFile(data_dir, 'r') as zip_ref:
        # Iterate over files in the zip archive
        for filename in tqdm(zip_ref.namelist()):
            # Check if the file is an image (you might need to adjust this condition)
            if filename.endswith(('.jpg', '.jpeg', '.png')): 
                label = 1 if "dog" in filename else 0  # Label: 1 for dog, 0 for cat

                # Read the image from the zip archive
                with zip_ref.open(filename) as img_file:
                    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
                    
                # Preprocess the image
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                images.append(img.flatten())  # Flatten the image into a 1D vector
                labels.append(label)
                

    return np.array(images), np.array(labels)

# Load dataset
print("Loading data...")
X, y = load_data(data_dir)
print("Data loaded.")

# Normalize pixel values
X = X / 255.0

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
print("Training the SVM model...")
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)
print("Model training completed.")

# Evaluate the model
print("Evaluating the model...")
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# Save the model
import joblib
joblib.dump(svm, "svm_dogs_vs_cats_model.pkl")
print("Model saved as svm_dogs_vs_cats_model.pkl.")
