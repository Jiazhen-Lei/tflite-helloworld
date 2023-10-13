# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""hello_world model training for sinwave recognition

Run:
`bazel build tensorflow/lite/micro/examples/hello_world:train`
`bazel-bin/tensorflow/lite/micro/examples/hello_world/train --save_tf_model --save_dir=/tmp/model_created/`
"""
import math
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 500, "number of epochs to train the model.")
# DEcade
flags.DEFINE_string("save_dir", "./models",
                    "the directory to save the trained model.")
# flags.DEFINE_string("save_dir", "/tmp/model_created/",
#                     "the directory to save the trained model.")
flags.DEFINE_boolean("save_tf_model", False,
                     "store the original unconverted tf model.")
flags.DEFINE_boolean("quantization", True,
                     "quantization or not")


def get_data():
    """
    The code will generate a set of random `x` values,calculate their sine
    values.
    """
    # Generate a uniformly distributed set of random numbers in the range from
    # 0 to 2π, which covers a complete sine wave oscillation
    x_values = np.random.uniform(low=0, high=2 * math.pi,
                                size=1000).astype(np.float32)

    # Shuffle the values to guarantee they're not in order
    # 随机排列
    np.random.shuffle(x_values)

    # Calculate the corresponding sine values
    y_values = np.sin(x_values).astype(np.float32)
    # # DEcade: Add a small random number to each y value
    # y_values += 0.1 * np.random.randn(*y_values.shape) 

    return (x_values, y_values)


def create_model() -> tf.keras.Model:
    model = tf.keras.Sequential()

    # First layer takes a scalar input and feeds it through 16 "neurons". The
    # neurons decide whether to activate based on the 'relu' activation function.
    # Dense全连接层
    model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(1, )))

    # The new second and third layer will help the network learn more complex
    # representations
    model.add(tf.keras.layers.Dense(16, activation='relu'))

    # Final layer is a single neuron, since we want to output a single value
    model.add(tf.keras.layers.Dense(1))

    # Compile the model using the standard 'adam' optimizer and the mean squared
    # error or 'mse' loss function for regression.
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Print a summary of the model's architecture
    # 打印模型结构
    # model.summary()

    return model


def convert_tflite_model(model):
    """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format
        Args:
            model (tf.keras.Model): the trained hello_world Model
        Returns:
            The converted model in serialized format.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model


def save_tflite_model(tflite_model, save_dir, model_name):
    """save the converted tflite model
    Args:
        tflite_model (binary): the converted model in serialized format.
        save_dir (str): the save directory
        model_name (str): model name to be saved
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_name)
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    logging.info("Tflite model saved to %s", save_dir)


def train_model(epochs, x_values, y_values):
    """Train keras hello_world model
        Args: epochs (int) : number of epochs to train the model
            x_train (numpy.array): list of the training data
            y_train (numpy.array): list of the corresponding array
        Returns:
            tf.keras.Model: A trained keras hello_world model
    """
    model = create_model()
    # training
    history = model.fit(x_values,
                        y_values,
                        epochs=epochs,
                        validation_split=0.2,
                        batch_size=64,
                        verbose=2)# 输出一行记录

    if FLAGS.save_tf_model:
        model.save(FLAGS.save_dir, save_format="tf")
        logging.info("TF model saved to %s", FLAGS.save_dir)

    return model, history

# From: DEcade
# plot history
def show_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'g.', label = 'Training loss')
    plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def convert_quantized_tflite_model(model, x_values):
    """Convert the save TF model to tflite model, then save it as .tflite
        flatbuffer format

        Args:
            x_train (numpy.array): list of the training data

        Returns:
            The converted model in serialized format.
    """

    # Convert the model to the TensorFlow Lite format with quantization
    def representative_dataset(num_samples=500):
        for i in range(num_samples):
            yield [x_values[i].reshape(1, 1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # # decade delete int8
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()
    return tflite_model

def main(_):
    x_values, y_values = get_data()
    trained_model, history = train_model(FLAGS.epochs, x_values, y_values)

    # Show history
    show_history(history)

    # Convert and save the model to .tflite
    tflite_model = convert_tflite_model(trained_model)
    save_tflite_model(tflite_model,
                        FLAGS.save_dir,
                        model_name="hello_world_float.tflite")
    if FLAGS.quantization:
        quantized_tflite_model = convert_quantized_tflite_model(trained_model, x_values)
        save_tflite_model(quantized_tflite_model,
                        FLAGS.save_dir,
                        model_name="hello_world_quantization.tflite")
        
if __name__ == "__main__":
  app.run(main)