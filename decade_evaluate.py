import os
import tensorflow as tf
from absl import app
from absl import flags
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.platform import resource_loader
import math

_PREFIX_PATH = resource_loader.get_path_to_datafile('')

def invoke_tflite_interpreter(input_shape, interpreter, x_value, input_index,
                              output_index):
    input_data = np.reshape(x_value, input_shape)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_index)
    y_quantized = np.reshape(tflite_output, -1)[0]
    return y_quantized


# Generate a list of 1000 random floats in the range of 0 to 2*pi.
def generate_random_int8_input(sample_count=200):
    # Generate a uniformly distributed set of random numbers in the range from
    # 0 to 2π, which covers a complete sine wave oscillation
    np.random.seed(42)
    x_values = np.random.uniform(low=0, high=2 * np.pi,
                                size=sample_count).astype(np.int8)
    return x_values


# Generate a list of 1000 random floats in the range of 0 to 2*pi.
def generate_random_float_input(sample_count=200):
    # Generate a uniformly distributed set of random numbers in the range from
    # 0 to 2π, which covers a complete sine wave oscillation
    np.random.seed(42)
    x_values = np.random.uniform(low=0, high=2 * np.pi,
                                size=sample_count).astype(np.float32)
    return x_values

# Invoke the tflite interpreter with x_values in the range of [0, 2*PI] and
# returns the prediction of the interpreter.
def get_tflite_prediction(model_path, x_values):
    # TFLite interpreter
    tflite_interpreter = tf.lite.Interpreter(
        model_path=model_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.
        BUILTIN_REF,
    )
    tflite_interpreter.allocate_tensors()

    input_details = tflite_interpreter.get_input_details()[0]
    output_details = tflite_interpreter.get_output_details()[0]
    input_shape = np.array(input_details.get('shape'))

    y_predictions = np.empty(x_values.size, dtype=np.float32)

    for i, x_value in enumerate(x_values):
        y_predictions[i] = invoke_tflite_interpreter(
            input_shape,
            tflite_interpreter,
            x_value,
            input_details['index'],
            output_details['index'],
        )
    return y_predictions

def print_size():
    basic_model_size = os.path.getsize("models/hello_world_float.tflite")
    quantization_model_size = os.path.getsize("models/hello_world_quantization.tflite")
    difference = basic_model_size - quantization_model_size
    print("Basic model is %d bytes" % basic_model_size)
    print("Quantized model is %d bytes" % quantization_model_size)
    print("Difference is %d bytes" % difference) 

def main(_):
    model_path = os.path.join(_PREFIX_PATH, 'models/hello_world_float.tflite')
    model_quantization_path = os.path.join(_PREFIX_PATH, 'models/hello_world_quantization.tflite')
    # float32
    x_values = generate_random_float_input()
    # Calculate the corresponding sine values
    y_true_values = np.sin(x_values).astype(np.float32)

    y_predictions = get_tflite_prediction(model_path, x_values)
    y_predictions_quantization = get_tflite_prediction(model_quantization_path, x_values)
    plt.plot(x_values, y_predictions, 'b.', label='TFLite Prediction')
    plt.plot(x_values, y_true_values, 'r.', label='Actual values')
    plt.plot(x_values, y_predictions_quantization, 'g.', label='TFLite Quantization Prediction')
    plt.legend()
    plt.show()
    print_size()

if __name__ == '__main__':
    app.run(main)
