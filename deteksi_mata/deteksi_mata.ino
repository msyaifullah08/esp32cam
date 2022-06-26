#define relayOn       LOW
#define pinRelay       2

#include <TensorFlowLite_ESP32.h>

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "mata_model_data.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 70 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup() {

  while (!Serial);
  Serial.begin(115200);

  pinMode(4, OUTPUT);
  pinMode(33, OUTPUT);
  digitalWrite(pinRelay, !relayOn);
  pinMode(pinRelay, OUTPUT);

  static tflite::ops::micro::AllOpsResolver resolver;

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_mata_model_data);

  Serial.print("model->version()=");
  Serial.println(model->version());

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model provided is schema version %d not equal "
      "to supported version %d.",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  
  micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
    tflite::ops::micro::Register_DEPTHWISE_CONV_2D()
  );

  micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_CONV_2D,
    tflite::ops::micro::Register_CONV_2D()
  );

  micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_AVERAGE_POOL_2D,
    tflite::ops::micro::Register_AVERAGE_POOL_2D());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  digitalWrite(4, HIGH);
}

void loop() {
  // Get image from provider.

//  digitalWrite(4, HIGH);
//  delay(100);

  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.uint8)) {
    error_reporter->Report("Image capture failed.");
  }
//  digitalWrite(4, LOW);


  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    error_reporter->Report("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  uint8_t skorMata = output->data.uint8[kPersonIndex];
  uint8_t skorTidakAdaMata = output->data.uint8[kNotAPersonIndex];
  RespondToDetection(error_reporter, skorMata, skorTidakAdaMata);

  if ((skorMata > 100) && (skorTidakAdaMata < 50))
  {
    digitalWrite(pinRelay, relayOn);
    digitalWrite(33, HIGH);
  }
  else
  {
    digitalWrite(pinRelay, !relayOn);
    digitalWrite(33, LOW);
  }

  delay(1000);
}
