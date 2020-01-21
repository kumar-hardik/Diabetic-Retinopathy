import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras.layers import *
from keras.applications import *
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.optimizers import *
from keras.activations import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm

np.random.seed(2019)

#Importing images and dataset
train_dir = '/project/DRDLM/hardik/Kaggle_data/resized_train_15'

train_df = pd.read_csv('/project/DRDLM/hardik/Kaggle_data/trainLabels15.csv')

train_df.rename(columns={"image": "id_code", "level": "diagnosis"}, inplace=True)

#Sampling 0 Class Images
train_df_review = train_df[train_df['diagnosis']>0]
temp = train_df[train_df['diagnosis']==0].sample(10000, replace=True)
train_df_review = pd.concat([train_df_review,temp], ignore_index=True)
train_df = train_df_review

train_df['diagnosis'].hist()
train_df['diagnosis'].value_counts()

# We will resize the images to 299x299, then create a single numpy array to hold the data.
def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def preprocess_image(image_path, desired_size=299):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im

N = train_df.shape[0]
x_train = np.empty((N, 299, 299, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train[i, :, :, :] = preprocess_image(f'/project/DRDLM/hardik/Kaggle_data/resized_train_15/{image_id}.jpg')

# Getting Dummies for diagnosis
y_train = pd.get_dummies(train_df['diagnosis']).values

print(x_train.shape)
print(y_train.shape)

# Creating Multilabels for the diagnosis levels
y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))

# Splitting the data in training and validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train_multi, test_size=0.10, random_state=2019)

# Data Generator
BATCH_SIZE = 32

datagen = ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        fill_mode='constant',   # set mode for filling points outside the input boundaries
        shear_range=0.1,
        cval=0., # value used for fill_mode = "constant"
        rotation_range=0.2, 
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

data_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)

# Kappa Score Callback and Model Weight save
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('/home/sgar2405/Hardik/model_weights/xception_2015_pretrain.h5')

        return

# Installing EfficientNet 
"""import subprocess
import sys

try:
    import efficientnet.keras as efn
except ImportError:
    subprocess.call([sys.executable, "-m", "pip", "install", "-U", "--pre", 'efficientnet'])
finally:
    import efficientnet.keras as efn"""


# Creating Xception Model Architecture
base_model = Xception(
    weights=None,
    include_top=False,
    input_shape=(299,299,3))

x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='sigmoid')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
output_layer = Dense(5, activation='sigmoid', name="Output_Layer")(x)
model = Model(base_model.input, output_layer)

model.load_weights('/home/sgar2405/Hardik/model_weights/xception_2015test_2.h5')
# Training last 5  layers with pretrained weights

for layer in model.layers[:-5]:
    layer.trainable = True

# Compling Model
model.compile(
        loss='binary_crossentropy', # Regression problem (Single output)
        optimizer=Adam(lr=0.000005),
        metrics=['accuracy']
    )

# Fitting Model
kappa_metrics = Metrics()

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[kappa_metrics])

# Plotting Plot of Accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label="Training_Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation_Accuracy")
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('xception_2015.png', dpi=100)
