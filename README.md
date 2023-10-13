# Hello World Example

This example is designed to demonstrate the absolute basics of using [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers).It includes the full end-to-end workflow of training a model, converting it for use with TensorFlow Lite for Microcontrollers for running inference on a microcontroller.

## Run decade_train.py

update：

1. add model_summary()
2. add show_history(history)
3. add quantization (float16) operation

run：

```
python decade_train.py
```

## Run decade_evaluate.py

update：

1. only retain tflite_interpreter
2. add plot prediction result
3. add print_size()	

```
python decade_evaluate.py
```

