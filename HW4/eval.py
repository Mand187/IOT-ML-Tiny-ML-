import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Define parameters
IMG_SIZE = 128  # Ensure this matches the training image size
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def load_images_from_directory(directory, label):
    images, labels = [], []
    for img_file in os.listdir(directory):
        if img_file.endswith(IMAGE_EXTENSIONS):
            img_path = os.path.join(directory, img_file)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return images, labels

def load_data(data_dir):
    pos_images, pos_labels = load_images_from_directory(os.path.join(data_dir, 'positive'), 1)
    neg_images, neg_labels = load_images_from_directory(os.path.join(data_dir, 'negative'), 0)
    
    images = np.array(pos_images + neg_images) / 255.0  # Normalize to [0,1]
    labels = np.array(pos_labels + neg_labels)
    
    return images, labels

def evaluate_model(model, images, labels):
    """Evaluates the model and prints performance metrics."""
    print("Evaluating model...")
    loss, accuracy = model.evaluate(images, labels, verbose=1)
    predictions = model.predict(images)
    pred_classes = (predictions > 0.5).astype("int32").flatten()
    
    precision = precision_score(labels, pred_classes)
    recall = recall_score(labels, pred_classes)
    conf_matrix = confusion_matrix(labels, pred_classes)
    
    print(f"\nTest Results:\n-------------------")
    print(f"Number of test images: {len(images)}")
    print(f"Positive images: {np.sum(labels)}")
    print(f"Negative images: {len(labels) - np.sum(labels)}")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

def display_individual_predictions(images, labels, pred_classes):
    print("\nIndividual Predictions:")
    for i, (true_label, pred_label) in enumerate(zip(labels, pred_classes)):
        true_class = "Positive" if true_label == 1 else "Negative"
        pred_class = "Positive" if pred_label == 1 else "Negative"
        result = "Correct" if true_label == pred_label else "Wrong"
        print(f"Image {i+1}: True: {true_class}, Predicted: {pred_class}, {result}")

def main():
    data_dir = 'dataset'  # Set dataset directory
    model_path = 'model.h5'
    
    print("Loading model...")
    model = load_model(model_path)
    
    print("Loading data...")
    images, labels = load_data(data_dir)
    
    evaluate_model(model, images, labels)
    pred_classes = (model.predict(images) > 0.5).astype("int32").flatten()
    
    display_individual_predictions(images, labels, pred_classes)

if __name__ == "__main__":
    main()