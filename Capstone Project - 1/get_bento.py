import bentoml
from tensorflow import keras

model = keras.models.load_model("model/xception_only_lr.h5")
bentoml.keras.save_model("kitchenware-classifier", model)

