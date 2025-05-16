
---

# Object Detection on Arduino Nano 33 BLE Sense

## Overview

This project focuses on building a **visual object detection system** for deployment on an **Arduino Nano 33 BLE Sense** using its onboard camera. The task involves training a neural network to detect the presence of a **custom target object**  and deploying the model using TensorFlow Lite for Microcontrollers.


## Repository Structure

```
project-root/
├── data_collection/ # Python code for gathering images using the Arduino's built in camera
├── training/        # Python code for dataset prep, training, TFLite conversion
├── embedded/        # Arduino C++ code for deployment
└── report.pdf       # Final report submitted
```

---

## Project Goals

* Detect a **custom object** (e.g., car, dog, shoe, etc.) using an image-based model.
* Train and optimize a CNN suitable for edge deployment.
* Deploy the model on an Arduino with minimal latency.
* Evaluate real-world performance (accuracy, inference speed, robustness).

---

## training/

Contains:

* Data loading and preprocessing scripts
* Model architecture (modified `vww_model.py`)
* Training pipeline (Keras/TensorFlow)

---

## embedded/

Contains:

* C++ sketch for Arduino IDE
* Includes TensorFlow Lite Micro runtime


## Model Architecture

* Based on: [MLPerf Tiny VWW Model](https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/vww_model.py)
* Modifications:

  * Changed final classifier to detect \[custom class]
  * Adjusted number of filters and depth to fit memory limits
  * Input image size: 96x96 grayscale

---

## Dataset and Training

* **Data Collection**: Captured using Arduino camera and smartphone
* **Augmentations**:

  * Rotation, flipping, brightness scaling
* **Data Sources**:

  * Custom captures
  * Public datasets: \[mention if used]
* **Training Strategy**:

  * Fine-tuning pretrained VWW model
  * Used quantization-aware training for accuracy post-TFLite
  * Batch size: XX, Epochs: XX, Optimizer: Adam

---

## Deployment on Arduino

* Converted trained model to `.tflite`
* Used `xxd` to embed model as C array (`model_data.cc`)
* Inference run inside `loop()` function with camera capture
* Visual feedback via Serial/LED

---


---

## Lessons Learned

* Model quantization significantly reduced accuracy; solved with QAT.
* Arduino memory constraints required aggressive model slimming.
* Real-time performance is feasible but highly dependent on image preprocessing.

---

## Conclusion

This project demonstrates a complete edge-AI pipeline—from data gathering to embedded inference—for custom object detection. The system works reliably in real-time for the target object and shows potential for practical applications in low-power vision tasks.

---

## References

* MLPerf Tiny Benchmark: [https://github.com/mlcommons/tiny](https://github.com/mlcommons/tiny)
* Arduino Person Detection Example: [https://github.com/tinyMLx/arduino-library/tree/main/examples/person\_detection](https://github.com/tinyMLx/arduino-library/tree/main/examples/person_detection)
* Visual Wake Words Model: [https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual\_wake\_words/vww\_model.py](https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/vww_model.py)
* TensorFlow Lite for Microcontrollers: [https://www.tensorflow.org/lite/microcontrollers](https://www.tensorflow.org/lite/microcontrollers)

---
