import os
import numpy as np

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xcep
import bentoml



def load_data(csv_file, path='data/images/'):
  train = pd.read_csv(csv_file)
  train['filename'] = path + train['Id'].map(lambda x: f'{x:0>4}.jpg').astype(str)
  return train

def split_data(data, cuttoff_percentage=0.8): 
  cutoff = int(len(train) * cuttoff_percentage)
  df_train = train[:cutoff]
  df_val = train[cutoff:]
  return df_train, df_val

def data_generator(preprocessing_function, df, y_col='label', class_mode='categorical', path='data/images/') 
  data_generator = ImageDataGenerator(preprocessing_function=preprocessing_function)   
  df_generator = data_generator.flow_from_dataframe(
          dataframe=df,
          directory=path,
          x_col='filename',
          y_col=y_col,
          target_size=(300, 300),
          batch_size=32,
          class_mode=class_mode,
      )
  return df_generator

def train_model():
  base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(299, 299, 3)
)
  base_model.trainable = False
  inputs = keras.Input(shape=(300, 300, 3))
  base = base_model(inputs, training=False)
  vectors = keras.layers.GlobalAveragePooling2D()(base)
  outputs = keras.layers.Dense(6)(vectors)
  model = keras.Model(inputs, outputs)
  optimizer = keras.optimizers.Adam(learning_rate=0.01)
  loss = keras.losses.CategoricalCrossentropy(from_logits=True)
  model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=['accuracy']
  )
  history = model.fit(train_generator, epochs=5, validation_data=validation_generator)
  return model



def main(train_csv, test_csv):
  df_train_full= load_data(train_csv)
  df_train, df_val = split_data(training_df)
  training_generator = data_generator(preprocess_input_xcep, df_train, y_col='label', class_mode='categorical')
  validation_generator = data_generator(preprocess_input_xcep, df_val, y_col='label', class_mode='categorical')
  model = train_model()
  save_model = bentoml.keras.save_model("kitchenware-classifier",
                                       model)

  

if __name__ == '__main__':
  train_csv = os.getenv('train_csv')
  test_csv = os.getenv('test_csv')
  main(train_csv, test_csv)
