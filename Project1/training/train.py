import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Reshape, Activation, Flatten, BatchNormalization,
                                     Conv2D, MaxPooling2D)
from tensorflow.keras.regularizers import l2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define parameters
IMG_SIZE_HEIGHT = 64
IMG_SIZE_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 2
LEARNING_RATE = 0.00015
L2_REG = 1e-4

def loadImagesFromDirectory(directory, label):
    images, labels = [], []
    for imgFile in os.listdir(directory):
        if imgFile.endswith(('.jpg', '.jpeg', '.png')):
            imgPath = os.path.join(directory, imgFile)
            img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH))
            imgArray = tf.keras.preprocessing.image.img_to_array(img)
            images.append(imgArray)
            labels.append(label)
    return images, labels

def loadData(dataDir):
    posImages, posLabels = loadImagesFromDirectory(os.path.join(dataDir, 'positive'), 1)
    negImages, negLabels = loadImagesFromDirectory(os.path.join(dataDir, 'negative'), 0)
    
    images = np.array(posImages + negImages) / 255.0  # Normalize
    labels = np.array(posLabels + negLabels)
    return images, labels

def buildCnnModel():
    # Model parameters
    input_shape = [IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, 3] 
    num_filters = 16 

    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten layer - FIXED: Use Flatten instead of Reshape
    x = Flatten()(x)
    
    # Dense layers
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plotTrainingHistory(history, outputDir):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir, 'trainingHistory.png'))

def convertToTflite(model, output_path):
    # FIXED: Proper implementation of int8 quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Define a representative dataset generator function
    def representative_dataset_gen():
        # Use a small subset of your training data here
        # This is just a placeholder - you should use actual data
        for _ in range(100):
            # Generate dummy data matching your input shape
            data = np.random.random((1, IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, 3))
            yield [data.astype(np.float32)]
    
    # Set the quantization configuration
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

def saveModelAndMetrics(model, history, x_train, y_train, x_val, y_val, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, 'model.h5'))
    
    try:
        convertToTflite(model, os.path.join(output_dir, 'model.tflite'))
    except Exception as e:
        print(f"Error converting to TFLite: {str(e)}")
        print("Saving without TFLite conversion")

    # Evaluate model
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

    # Save metrics
    with open(os.path.join(output_dir, 'model_metrics.txt'), 'w') as f:
        f.write(f"Training Accuracy: {train_acc:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Positive Samples: {np.sum(y_train) + np.sum(y_val)}\n")
        f.write(f"Negative Samples: {len(y_train) + len(y_val) - np.sum(y_train) - np.sum(y_val)}\n")

    plotTrainingHistory(history, output_dir)

def main():
    dataDir = os.path.join(os.path.dirname(__file__), 'dataset')
    outputDir = os.path.join(os.path.dirname(__file__), 'Output')
    
    # Load and split data
    print("Loading data...")
    images, labels = loadData(dataDir)
    xTrain, xVal, yTrain, yVal = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
    
    print(f"Training samples: {len(xTrain)}, Validation samples: {len(xVal)}")
    
    # Build and train model
    print("Building model...")
    model = buildCnnModel()
    model.summary()
    
    print("Training model...")
    history = model.fit(xTrain, yTrain, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(xVal, yVal), verbose=1)
    
    # Save results
    saveModelAndMetrics(model, history, xTrain, yTrain, xVal, yVal, outputDir)
    print(f"Model and metrics saved in {outputDir}")

if __name__ == "__main__":
    main()