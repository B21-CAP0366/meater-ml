# meater-ml

This repository is for Meater Machine Learning model.

## Getting Started
1. Open [meater.ipynb](meater.ipynb) with Google Colaboratory.
1. It's better to "Save a copy in Drive" to run and edit with your account. Click `File` > `Save a copy in Drive` in your Google Colaboratory.
1. Create `kaggle.json` from Kaggle. [Learn how to make it](https://github.com/Kaggle/kaggle-api#api-credentials).
1. Run the colab and it will ask you to upload your `kaggle.json`.
1. It will automatically download the [dataset](https://www.kaggle.com/crowww/meat-quality-assessment-based-on-deep-learning) from Kaggle.
1. Done. You can explore the colaboratory.

## How to Use Saved Model

### TensorFlow Lite

Steps:
1. Load the [TFLite model](model/1/converted_model.tflite)
    ```python
    with open(tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()
    ```
1. Make an interpreter
    ```python
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    ```
1. Make input and output index 
    ```python
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    ```
1. Load the test image
    ```python
    img = tf.keras.preprocessing.image.load_img(test_path, target_size=(dim, dim))
    ```
1. Convert the image to array
    ```python
    x = tf.keras.preprocessing.image.img_to_array(img)
    ```
1. Make to 4-dimensional array
    ```python
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    ```
1. Set tensor in interpreter (Input image)
    ```python
    interpreter.set_tensor(input_index, images)
    ```
1. Predict the image by invoke the interpreter
    ```python
    interpreter.invoke()
    ```
1. Get tensor in interpreter (Get predicted value)
    ```python
    interpreter.get_tensor(output_index)
    ```