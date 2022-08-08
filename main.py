
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import joblib
from tensorflow.keras import metrics, layers, models, optimizers, preprocessing, applications, utils
from random import randint
import os
import json
import csv
import custom_loader as cl

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

# SETTING PARAMETERS

seed = 928347
tam = 396
image_size = (tam, tam)
batch_size = 32
shape = (tam, tam, 3)
epochs = 50
validation_split = .05
class_names = ["NORMAL", "CNV", "DME", "DRUSEN"]

# GENERATE EXPERIMENT

modelos = {
	"Inc" : applications.InceptionV3(weights="imagenet", include_top=False, input_shape=shape),
	"Res" : applications.ResNet50(weights="imagenet", include_top=False, input_shape=shape),
	"Vgg" : applications.VGG16(weights="imagenet", include_top=False, input_shape=shape)
	}

def adjust_img(image, label):
	return image/255, label

for model_name, base in modelos.items():	

	print(model_name)

	#LOADING DATASET

	train, val = cl.customDataset(
		"/home/mauricio/dados/Mauricio/OCT2017/test", 
		validation_split=validation_split, 
		label_names=class_names, 
		batch_size=batch_size,
		h=tam, 
		w=tam,
		resize="interpolation"
		)

	test, _ = cl.customDataset(
		"/home/mauricio/dados/Mauricio/OCT2017/test", 
		label_names=class_names, 
		batch_size=batch_size,
		h=tam, 
		w=tam,
		resize="interpolation"
		)

	train = train.map(adjust_img)
	val = val.map(adjust_img)
	test = test.map(adjust_img)

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

	learning_hist = jarbas.fit(train, epochs=epochs, validation_data=val)

	# TESTING	
	
	con_mat = []
	con_mat_tensor = tf.math.confusion_matrix(np.concatenate([label for _,label in test]), np.argmax(jarbas.predict(test), axis=-1))
	for i in con_mat_tensor:
		con_mat.append(list(i.numpy()))

	print(f"{model_name}NO FINE TUNE"}
	jarbas.evaluate(teste)

	#SAVING
	
	dir_name = model_name+"0/"

	try:
		os.mkdir(dir_name)
	except:
		pass

	with open(dir_name+'learning_hist.json', 'w') as f:
		json.dump(learning_hist.history, f)
	
	jarbas.save(dir_name+"saved_model")

	with open(dir_name+"confusion_matrix.csv", 'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerows(con_mat)
	
	# FINE TUNING

	jarbas.layers[0].trainable = True

	jarbas.compile(
		loss="sparse_categorical_crossentropy", 
		optimizer="Adam", 
		metrics=tfa.metrics.CohenKappa(4, sparse_labels=True))

	learning_hist = jarbas.fit(train, epochs=epochs//10, validation_data=val)

	# TESTING FINE TUNE
		
	con_mat = []
	con_mat_tensor = tf.math.confusion_matrix(np.concatenate([label for _,label in test]), np.argmax(jarbas.predict(test), axis=-1))
	for i in con_mat_tensor:
		con_mat.append(list(i.numpy()))

	print(f"{model_name}FINE TUNE"}
	jarbas.evaluate(teste)

	#SAVING FINE TUNE
	
	dir_name = model_name+"1/"

	try:
		os.mkdir(dir_name)
	except:
		pass

	with open(dir_name+'learning_hist.json', 'w') as f:
		json.dump(learning_hist.history, f)
	
	jarbas.save(dir_name+"saved_model")

	with open("confusion_matrix.csv", 'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerows(con_mat)
	
