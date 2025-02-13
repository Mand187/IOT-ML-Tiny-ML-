import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import person
from sklearn.datasets import load_iris
import tensorflow as tf

rng = np.random.default_rng(2022)

# Define data
listOfNames = ['Roger', 'Mary', 'Luisa', 'Elvis']
listOfAges = [23, 24, 19, 86]
listOfHeightsCm = [175, 162, 178, 182]

for name in listOfNames:
    print("The name {:} is {:} letters long".format(name, len(name)))

# List comprehension to make a list of the lengths of the names
lengthsOfNames = [len(name) for name in listOfNames]
print(lengthsOfNames)

# Create a dictionary named "people"
people = {}

# Iterate through the names and create Person objects
for name, age, height in zip(listOfNames, listOfAges, listOfHeightsCm):
    createdPerson = person.Person(name, age, height)
    people[name] = createdPerson

# Convert the lists to NumPy arrays
agesArray = np.array(listOfAges)
heightsArray = np.array(listOfHeightsCm)

# Calculate the average age
averageAge = np.mean(agesArray)

# Create a scatter plot
plt.scatter(agesArray, heightsArray, label='People')
plt.grid(True)
plt.xlabel('Ages')
plt.ylabel('Heights (cm)')
plt.title('Scatter Plot of Ages vs Heights')
plt.legend()
plt.savefig('scatterPlot.png')
plt.show()

# Load iris dataset
irisDb = load_iris(as_frame=True)
xData = irisDb['data']
yLabels = irisDb['target']
targetNames = irisDb['target_names']

# Plot iris data
fig = plt.figure(figsize=(6, 6), dpi=100, facecolor='w', edgecolor='k')
colors = ['maroon', 'darkgreen', 'blue']
for n, species in enumerate(targetNames):
    plt.scatter(xData[yLabels == n].iloc[:, 0], xData[yLabels == n].iloc[:, 1], c=colors[n], label=targetNames[n])
plt.xlabel(irisDb['feature_names'][0])
plt.ylabel(irisDb['feature_names'][1])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('irisData.png')

# Define classifiers
def classifyRand(x):
    return rng.integers(0, 3, endpoint=False)

def classifyIris(features):
    weights = np.array([[0.7230, -0.5053, -0.8363, 0.1594],
                        [0.4765, -0.9116, 0.4933, -0.7420],
                        [0.9793, -2.5114, 0.3961, 0.5976]])
    biases = np.array([0.1, 0.2, 0.3])
    scores = np.matmul(weights, features) + biases
    label = int(np.argmax(scores))
    return label

def evaluateClassifier(classifierFunction, xData, labels, printConfusionMatrix=True):
    nCorrect = 0
    nTotal = xData.shape[0]
    cm = np.zeros((3, 3))
    for i in range(nTotal):
        x = xData[i, :]
        y = classifierFunction(x)
        yTrue = labels[i]
        cm[yTrue, y] += 1
        if y == yTrue:
            nCorrect += 1
    accuracy = nCorrect / nTotal
    global acc
    acc = accuracy  # Store accuracy for test cases
    print(f"Accuracy = {nCorrect} correct / {nTotal} total = {100.0 * accuracy:3.2f}%")
    if printConfusionMatrix:
        print(f"{' ' * 12}Estimated Labels")
        print(f"{' ' * 14}{0:3.0f}  {1.0:3.0f}  {2.0:3.0f}")
        print(f"{' ' * 12}{'-' * 15}")
        for i in range(3):
            print(f"True {i:5.0f} |   {cm[i, 0]:3.0f}  {cm[i, 1]:3.0f}  {cm[i, 2]:3.0f} ")
        print(f"{'-' * 40}")
    return accuracy, cm

# Evaluate classifiers
evaluateClassifier(classifyRand, xData.to_numpy(), yLabels.to_numpy())
evaluateClassifier(classifyIris, xData.to_numpy(), yLabels.to_numpy())

# Define TensorFlow model
tfModel = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

tfModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = tfModel.fit(xData.to_numpy(), yLabels.to_numpy(), epochs=100, batch_size=32, validation_split=0.2)

trainLoss, trainAccuracy = tfModel.evaluate(xData.to_numpy(), yLabels.to_numpy())
print("Training Loss:", trainLoss)
print("Training Accuracy:", trainAccuracy)
