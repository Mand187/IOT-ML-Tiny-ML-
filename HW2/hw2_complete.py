import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

def build_model1():
    model1 = models.Sequential([
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation="softmax")
    ])

    # Compile the model
    model1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model1.summary()

    return model1

def build_model2():
    model2 = models.Sequential([
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64, (3, 3), strides=(2, 2), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), strides=(2, 2), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation="softmax")
    ])

    # Compile the model
    model2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model2.summary()

    return model2

def build_model3():
    inputs = Input(shape=(32,32,3))
        
    residual = layers.Conv2D(32, (3, 3), strides=(2, 2), name='Conv1', activation = 'relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(residual)
    conv1 = layers.Dropout(0.5)(conv1)

    conv2 = layers.Conv2D(64, (3, 3), strides=(2, 2), name='Conv2', activation = 'relu', padding='same')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(0.5)(conv2)
    
    conv3 = layers.Conv2D(128, (3, 3), strides=(2, 2), name='Conv3',  activation = 'relu', padding='same')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Dropout(0.5)(conv3)

    skip1 = layers.Conv2D(128, (1,1), strides=(4, 4), name="Skip1")(residual)
    skip1 = layers.Add()([skip1, conv3])

    conv4 = layers.Conv2D(128, (3, 3), name='Conv4', activation = 'relu', padding='same')(skip1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Dropout(0.5)(conv4)

    conv5 = layers.Conv2D(128, (3, 3), name='Conv5', activation = 'relu', padding='same')(conv4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Dropout(0.5)(conv5)

    skip2 = layers.Add()([skip1, conv5])

    conv6 = layers.Conv2D(128, (3, 3), name='Conv6', activation = 'relu', padding='same')(skip2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Dropout(0.5)(conv6)

    conv7 = layers.Conv2D(128, (3, 3), name='Conv7', activation = 'relu', padding='same')(conv6)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Dropout(0.5)(conv7)

    skip3 = layers.Add()([skip2, conv7])
    
    pool = layers.MaxPooling2D((4, 4), strides=(4, 4))(skip3)
    flatten = layers.Flatten()(pool)
    
    dense = layers.Dense(128, activation = 'relu')(flatten)
    dense = layers.BatchNormalization()(dense)

    output = layers.Dense(10, activation = 'softmax')(dense)
    model3 = Model(inputs=inputs, outputs=output)

    model3.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model3.summary()

    return model3

def build_model50k():
    model50k = models.Sequential([
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation="softmax")
    ])
    
    # Compile the model
    model50k.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Print model summary
    model50k.summary()
    return model50k

def train_and_evaluate_model(model, model_name, train_images, train_labels, val_images, val_labels, test_images, test_labels, epochs=50):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels))

    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"{model_name} Test accuracy: {test_accuracy}")

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    print(f"{model_name} Final Training Accuracy: {train_accuracy[-1]}")
    print(f"{model_name} Final Validation Accuracy: {val_accuracy[-1]}")

    model.save(f"{model_name}.h5")
    print(f"{model_name} saved")

    loaded_model = tf.keras.models.load_model(f"{model_name}.h5")
    loaded_model.summary()
    print(f"\n{model_name} Saved")

    return loaded_model

if __name__ == "__main__":
 # Load and preprocess data
    (train_images_all, train_labels_all), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, val_images, train_labels, val_labels = train_test_split(train_images_all, train_labels_all, test_size=0.1, random_state=42)
    train_images = train_images.astype('float32') / 255
    val_images = val_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Train and evaluate models
    model1 = build_model1()
    model1 = train_and_evaluate_model(model1, "model1", train_images, train_labels, val_images, val_labels, test_images, test_labels)

    # Image prediction (using the loaded model1)
    img_path = 'test_image_IOT_ML.jpg'  # Make sure 'cat.jpg' is in the same directory
    try:
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model1.predict(img_array)
        predicted_class = np.argmax(predictions)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        predicted_class_name = class_names[predicted_class]
        print("Predicted Class:", predicted_class_name)
    except FileNotFoundError:
        print(f"Error: Image file '{img_path}' not found.")


    model2 = build_model2()
    model2 = train_and_evaluate_model(model2, "model2", train_images, train_labels, val_images, val_labels, test_images, test_labels)

    model3 = build_model3()
    model3 = train_and_evaluate_model(model3, "model3", train_images, train_labels, val_images, val_labels, test_images, test_labels)

    model50k = build_model50k()
    model50k = train_and_evaluate_model(model50k, "best_model", train_images, train_labels, val_images, val_labels, test_images, test_labels)
    