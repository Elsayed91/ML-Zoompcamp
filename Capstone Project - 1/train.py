import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficient




def load_data(csv_file):
  train = pd.read_csv(csv_file)
  train['filename'] = '/kaggle/input/kitchenware-classification/images/' + train['Id'].map(lambda x: f'{x:0>4}.jpg').astype(str)
  return train

def split_data(data, cuttoff_percentage=0.8): 
  cutoff = int(len(train) * cuttoff_percentage)
  df_train = train[:cutoff]
  df_val = train[cutoff:]
  return df_train, df_val

def data_generator(preprocessing_function, df, y_col='label', class_mode='categorical') 
  data_generator = ImageDataGenerator(preprocessing_function=preprocess_input_efficient)   
  df_generator = data_generator.flow_from_dataframe(
          dataframe=df,
          directory='/kaggle/input/kitchenware-classification/images/',
          x_col='filename',
          y_col=y_col,
          target_size=(300, 300),
          batch_size=32,
          class_mode=class_mode,
      )
  return df_generator

def train_efficient_net():
  base_model = EfficientNetB4(
      weights='imagenet',
      include_top=False,
      input_shape=(300, 300, 3)
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
  return model, history

def predict(train_generator, test_generator):
  y_pred = model.predict(test_generator)
  classes = np.array(list(train_generator.class_indices.keys()))
  return classes[y_pred.argmax(axis=1)]

def main(training_file, test_file):
  training_df = load_data(training_file)
  test_df = load_data(test_file)
  df_train, df_val = split_data(training_df)
  training_generator = data_generator(preprocess_input_efficient, df_train, y_col='label', class_mode='categorical')
  validation_generator = data_generator(preprocess_input_efficient, df_val, y_col='label', class_mode='categorical')
  test_generator = data_generator(preprocess_input_efficient, df_test, y_col=None, class_mode=None)
  predictions = predict(train_generator, test_generator)

  

if __name__ == '__main__':
  main(training_file, test_file)
