#include "model.h"
#include "mbed.h"

#include <Arduino_OV767X.h>
#include <TensorFlowLite.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

constexpr int frameSize = 176 * 144;
constexpr int tensorArenaSize = 1024 * 180;

static uint8_t frameData[frameSize];
static uint8_t *tensorArena = nullptr;

// TensorFlow Lite objects
static const tflite::Model* tfluModel = nullptr;
static tflite::MicroInterpreter* tfluInterpreter = nullptr;
static TfLiteTensor* tfluITensor = nullptr;
static TfLiteTensor* tfluOTensor = nullptr;
static float tfluScale = 0.0f;
static int32_t tfluZeroPoint = 0;


void captureFrame() {
    Camera.readFrame(frameData);
    for (int idx = 0; idx < frameSize; ++idx) {
        tfluITensor->data.int8[idx] = frameData[idx] - 128;
    }
}

void initializeTflu() {
    Serial.println("Initializing TensorFlow Lite...");
    
    tfluModel = tflite::GetModel(model_tflite);
    if (tfluModel->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version mismatch!");
        while (1);
    }
    
    tflite::AllOpsResolver tfluOpsResolver;
    tensorArena = new uint8_t[tensorArenaSize];
    
    static tflite::MicroInterpreter staticInterpreter(tfluModel, tfluOpsResolver, tensorArena, tensorArenaSize);
    tfluInterpreter = &staticInterpreter;
    
    if (tfluInterpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: Failed to allocate TFLu tensors!");
        while (1);
    }
    
    tfluITensor = tfluInterpreter->input(0);
    tfluOTensor = tfluInterpreter->output(0);
    
    auto* iQuant = reinterpret_cast<TfLiteAffineQuantization*>(tfluITensor->quantization.params);
    tfluScale = iQuant->scale->data[0];
    tfluZeroPoint = iQuant->zero_point->data[0];
    
    Serial.println("TensorFlow Lite initialized successfully.");
}


void runInference() {
    int startTime = millis();  // Start timer before inference
    
    if (tfluInterpreter->Invoke() != kTfLiteOk) {
        Serial.println("ERROR: invoking TFLu interpreter");
        return;
    }
    
    int elapsedTime = millis() - startTime;  // Measure after inference

    int8_t output = tfluOTensor->data.int8[0];

    // Print inference result
    Serial.print("Inference Result: ");
    if (output > 0) {
        Serial.println("Box");
    } else {
        Serial.println("No Box");
    }

    // Print confidence (raw or scaled)
    Serial.print("Confidence: ");
    Serial.println(output);

    // Print performance metrics
    Serial.print("Inference Time (ms): ");
    Serial.print(elapsedTime);
    Serial.print(" | Sample Rate (FPS): ");
    Serial.println(1000.0f / elapsedTime, 2);
}

void setup() {
    Serial.begin(115200);
    while (!Serial);
    
    if (!Camera.begin(QCIF, GRAYSCALE, 5)) {
        Serial.println("ERROR: Camera Failed to Init");
        while (1);
    }
    
    initializeTflu();
    
    Serial.println("Performing initial inference...");
    int startTime = millis();
    tfluInterpreter->Invoke();
    Serial.print("Inference Time: ");
    Serial.println(millis() - startTime);
}

void loop() {
    captureFrame();
    runInference();
}
