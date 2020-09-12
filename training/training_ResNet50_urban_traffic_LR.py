import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import csv
import pickle

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


def tensorflow_fix():
    """Call this function to solve the 'failed to get convolution algorithm error'
    """
    error = "Not enough GPU hardware devices available"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, error
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


tensorflow_fix()

ROOT_PATH= "/home/fabiogarcea/projects/waterview_zahra/"
train_path =ROOT_PATH+ "/data/data_urban_traffic/train/train.csv"
val_path = ROOT_PATH+"/data/data_urban_traffic/train/validation.csv"
test_path = ROOT_PATH+"/data/data_urban_traffic/test"

root_train_dir = ROOT_PATH+"/data/data_urban_traffic/train"

BATCH_SIZE = 32
TARGET_SIZE =(640, 480)
LEARNING_RATE = 0.0001
EPOCHS = 10
MODEL_DEST_PATH = ROOT_PATH + "models/resnet50_imagenet_pretrained_urban_traffic_LR.h5"
HIST_DEST_PATH = ROOT_PATH + "references/trainings/resnet50_imagenet_pretrained_urban_traffic_history_LR"


def number_of_element(path):
    input_file = open(path, "r+")
    reader_file = csv.reader(input_file)
    total = len(list(reader_file))
    return total


totalTrain = number_of_element(train_path)
totalVal = number_of_element(val_path)

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(val_path)

train_df["class"] = train_df["class"].astype(str)
valid_df["class"] = valid_df["class"].astype(str)

train_datagen = ImageDataGenerator(
    dtype='float32',
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

valid_generate = ImageDataGenerator(
    dtype='float32',
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=root_train_dir,
    x_col="filename",
    y_col="class",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

validation = valid_generate.flow_from_dataframe(
    dataframe=valid_df,
    directory=root_train_dir,
    x_col="filename",
    y_col="class",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

gen = ImageDataGenerator(dtype='float32',
    preprocessing_function=preprocess_input)
test = gen.flow_from_directory(
    directory=test_path, class_mode="categorical", target_size=TARGET_SIZE, batch_size=BATCH_SIZE)

base_model = ResNet50(include_top=False, weights='imagenet',
                      input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))


x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dense(2, activation='softmax', name='probs')(x)
model = Model(base_model.input, x)

model.compile(optimizer=Adam(lr=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("start training ...")
history = model.fit(train,
                              steps_per_epoch=totalTrain // BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_data=validation,
                              validation_steps=totalVal // BATCH_SIZE)


model.save(MODEL_DEST_PATH)
print("Saved model to disk")

with open(HIST_DEST_PATH, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)