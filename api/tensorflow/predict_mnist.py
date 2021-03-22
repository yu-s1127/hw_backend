import tensorflow as tf
import numpy as np

def predict(image, model_type):
    if model_type == "auged":
        model_name = "augmented_model"
    else:
        model_name = "simple_model"
    loaded_model = tf.keras.models.load_model(f"api/tensorflow/saved_models/{model_name}.h5")
    prediction = loaded_model.predict(image)
    predicted_label = np.argmax(prediction)
    prediction_prob = prediction[0].tolist()

    return predicted_label, prediction_prob