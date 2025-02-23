import os
import cv2
import numpy as np
from tensorflow import keras

from task2.utils.data_loader import processing_image
from task2.utils.paths import CV_MODEL_PATH


def inference_cv(data, size=(224, 224)):
    if isinstance(data, np.ndarray):
        return inference_cv_from_img(data, size)
    else:
        return inference_cv_from_img_path(data, size)


def inference_cv_from_img_path(img_path, size=(224, 224)):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    return inference_cv_from_img(img, size=size)


def preprocess_image(img, size=(224, 224)):
    # convert from bgr to rgb because cv2 reads image in bgr
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize to suitable size
    image = cv2.resize(image, size)
    # normalize pixels
    image = image / 255.0
    # loaded image has shape (224, 224, 3), we need (1, 224, 224, 3) where 1 is the batch size
    image = np.expand_dims(image, axis=0)
    # make sure its float
    image = image.astype(np.float32)
    return image


def inference_cv_from_img(img, size=(224, 224)):
    # preprocess image
    img = preprocess_image(img, size=size)

    vgg16 = keras.models.load_model(CV_MODEL_PATH)
    predictions = vgg16.predict(img)
    return predictions
