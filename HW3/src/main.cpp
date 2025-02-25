#include <Arduino.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <TinyMLShield.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "sine_model_data.h"

#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 128
#define INT_ARRAY_SIZE 8   

char received_char = (char)NULL;              
int chars_avail = 0;                    
char out_str_buff[OUTPUT_BUFFER_SIZE];  
char in_str_buff[INPUT_BUFFER_SIZE];    
int input_array[INT_ARRAY_SIZE];        
int in_buff_idx = 0; 
int array_length = 0;

int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);

#define TENSOR_ARENA_SIZE 2048
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void setup() {
  delay(5000);
  Serial.begin(115200);
  while (!Serial) { } /
  Serial.println("TFLM Sine Prediction Project waking up");

  // Initialize TFLM model
  model = tflite::GetModel(sin_model_data_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version %d not equal to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
    while (1);
  }
  
  // Set up the operations resolver and interpreter.
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;
  
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }
  
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  
  memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
}

void loop() {
  chars_avail = Serial.available();
  if (chars_avail > 0) {
    received_char = Serial.read();
    Serial.print(received_char);  
    in_str_buff[in_buff_idx++] = received_char;
    
    if (received_char == 13) {
      Serial.print("Processing line: ");
      Serial.println(in_str_buff);
      
      array_length = string_to_array(in_str_buff, input_array);
      
      if (array_length != 7) {
        Serial.println("Error: Please enter exactly 7 comma-separated integers.");
      } else {
        sprintf(out_str_buff, "Parsed integers: ");
        Serial.print(out_str_buff);
        print_int_array(input_array, array_length);
        
        for (int i = 0; i < 7; i++) {
          input_tensor->data.int8[i] = (int8_t) input_array[i];
        }
        
        unsigned long t0 = micros();
        Serial.println("Test print statement");
        unsigned long t1 = micros();
        
        if (interpreter->Invoke() != kTfLiteOk) {
          error_reporter->Report("Invoke failed.");
        }
        unsigned long t2 = micros();
        
        unsigned long t_print = t1 - t0;
        unsigned long t_infer = t2 - t1;
        
        int8_t prediction = output_tensor->data.int8[0];
        
        sprintf(out_str_buff, "Prediction: %d", prediction);
        Serial.println(out_str_buff);
        sprintf(out_str_buff, "Printing time (us): %lu", t_print);
        Serial.println(out_str_buff);
        sprintf(out_str_buff, "Inference time (us): %lu", t_infer);
        Serial.println(out_str_buff);
      }
      
      memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
      in_buff_idx = 0;
    }
    else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
      memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
      in_buff_idx = 0;
    }
  }
}

int string_to_array(char *in_str, int *int_array) {
  int num_integers = 0;
  char *token = strtok(in_str, ",");
  while (token != NULL && num_integers < INT_ARRAY_SIZE) {
    int_array[num_integers++] = atoi(token);
    token = strtok(NULL, ",");
  }
  return num_integers;
}

void print_int_array(int *int_array, int array_len) {
  Serial.print("[");
  for (int i = 0; i < array_len; i++) {
    Serial.print(int_array[i]);
    if (i < array_len - 1) {
      Serial.print(", ");
    }
  }
  Serial.println("]");
}
