
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras import layers, models, optimizers, preprocessing, applications, utils
from random import randint
import os
import json
import custom_loader as cl

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

# SETTING PARAMETERS

seed = 928347
tam = 396
image_size = (tam, tam)
batch_size = 128
shape = (tam, tam, 3)
epochs = 1
validation_split = .05
class_names = ["NORMAL", "CNV", "DME", "DRUSEN"]

# GENERATE EXPERIMENT

modelos = {
	"Inc" : applications.InceptionV3(weights="imagenet", include_top=False, input_shape=shape),
	"Res" : applications.ResNet50(weights="imagenet", include_top=False, input_shape=shape),
	"Vgg" : applications.VGG16(weights="imagenet", include_top=False, input_shape=shape)
	}

# With/Without fine tuning
fine_tune = ["Ft", "Wo"]

# Bilinear Interpolation/Zero Padding
resize = ["interpolation", "zeropadding"]

for model_name, base in modelos.items():
	#LOADING DATASET
	train, val = cl.customDataset(
		"/home/mauricio/dados/Mauricio/OCT2017/train", 
		validation_split=validation_split, 
		label_names=class_names, 
		batch_size=batch_size,
		h=tam, 
		w=tam,
		resize=resize[0]
		)

	# CREATING MODEL

	base.trainable = False

	jarbas = models.Sequential()
	jarbas.add(base)
	jarbas.add(layers.Flatten())
	jarbas.add(layers.Dense(4, activation='softmax'))

	jarbas.compile(
		loss="sparse_categorical_crossentropy", 
		optimizer="Adam", 
		metrics=tfa.metrics.CohenKappa(4, sparse_labels=True))

	# TRAINING MODEL

	jarbas.fit(train, epochs=epochs, validation_data=val)

	# TESTING	
	
	test, _ = cl.customDataset(
		"/home/mauricio/dados/Mauricio/OCT2017/test", 
		label_names=class_names, 
		batch_size=batch_size,
		h=tam, 
		w=tam,
		resize=resize[0]
		)

	#|||||||do fine tuning|||||||

