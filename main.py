
import tensorflow as tf
import numpy as np
from tensorflow.keras import applications, models, layers, optimizers, metrics
import custom_loader as cl

import os
from math import ceil
from random import getrandbits

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# SETTING CONFIG VALUES

dataset_dir = "OCT-DATABASE/Mult/"
batch_size = 16
size = 396
epochs = 50
fine_epochs = 10
validation_split = 0.05
random_seed = getrandbits(64)

# LOADING DATASET

train, val = cl.customDataset(dataset_dir+"train/", rand=random_seed ,validation_split=validation_split, h=size, w=size, batch_size=batch_size)
test, _ = cl.customDataset(dataset_dir+"test/", rand=False, h=size, w=size, batch_size=batch_size)

def func(image, label):
        return image/255, label

train = train.map(func)
val = val.map(func)
test = test.map(func)

# CREATING MODEL FOR TRAINING

jarbas = models.Sequential()
jarbas.add(applications.VGG16(weights="imagenet", include_top=False, input_shape=(size, size, 3)))
jarbas.add(layers.Flatten())
jarbas.add(layers.Dense(4, activation="softmax"))
jarbas.layers[0].trainable = False

jarbas.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="Adam",
        metrics=["accuracy"])

# NORMAL TRAINING

jarbas.fit(train, validation_data=val, epochs=epochs)

# SETTING UP FOR FINE TUNING

jarbas.layers[0].trainable = True

jarbas.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=10e-5),
        metrics=["accuracy"])

# FINE TUNING

jarbas.fit(train, validation_data=val, epochs=fine_epochs)

# TESTING

con_mat = []
con_mat_tensor = tf.math.confusion_matrix(np.concatenate([label for _,label in test]), np.argmax(jarbas.predict(test), axis=-1))
for i in con_mat_tensor:
        con_mat.append(list(i.numpy()))

print(*con_mat)
jarbas.evaluate(test)

# SAVING

jarbas.save("trained_jarbas")
