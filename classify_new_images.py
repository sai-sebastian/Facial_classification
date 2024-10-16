import numpy as np
from skimage.io import imread
from skimage.transform import resize
import joblib


def classify_image(image_path):
    # Load and preprocess the image
    image = imread(image_path, as_gray=True)
    image_resized = resize(image, (32, 32), anti_aliasing=True)
    image_flattened = image_resized.flatten().reshape(1, -1)  # Flatten the resized image

    # Load the trained models
    svm_clf = joblib.load('svm_model.pkl')
    mlp_clf = joblib.load('mlp_model.pkl')
    dt_clf = joblib.load('dt_model.pkl')

    # Perform classification
    svm_pred = svm_clf.predict(image_flattened)

    mlp_pred = mlp_clf.predict(image_flattened)
    dt_pred = dt_clf.predict(image_flattened)


    return svm_pred, mlp_pred, dt_pred


# Example usage
new_image_path = "test1.jpg"
classification_results = classify_image(new_image_path)
print("SVM Prediction:", classification_results[0])
print("MLP Prediction:", classification_results[1])
print("Decision Tree Prediction:", classification_results[2])
print("MLP Prediction:", classification_results[1])


