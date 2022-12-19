import os

import bentoml
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('model/EfficientNetB401_0.962.h5')
bentoml.keras.save_model("keras_effnet", model)
