
import numpy as np
import tensorflow as tf
import keras as kr

from random import shuffle
from time import time
from glob import glob

def getFilePaths(directory):
	return [file_path for file_path in glob(directory+"/*/*.jpeg")]

def getLabelsNames(directory):
	return [dir_name.split("/")[-1] for dir_name in glob(directory+"/*")] 

def getImageLabel(file_path, label_names, h, w, resize):
	img = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)
	resize = str(resize).split("'")[1]
	if resize=="interpolation":
		resize = lambda img, w, h: tf.image.resize(img, [w, h])
	elif resize=="zeropadding":
		resize = lambda img, w, h: tf.image.resize_with_pad(img, w, h)
	x = resize(img, w, h)
	
	y = 0
	for lb in label_names:
		if str(lb)[0] == "b":
			lb = str(lb).split("'")[1]
		if str(file_path).split("/")[-2] == lb:
			break
		y += 1
	return x, y

def loadImagesGen(file_paths, label_names, h, w, resize):
	for file_path in file_paths:
		yield getImageLabel(file_path, label_names, h, w, resize)

def customDataset(directory, rand=True, validation_split=0, label_names=None, batch_size=32, h=512, w=512, resize="interpolation"):
	file_paths = getFilePaths(directory)
	if rand:
		shuffle(file_paths)

	val = None	
	if validation_split >= 0:
		size_val = int(validation_split*len(file_paths))
		val_file_paths = file_paths[:size_val]
		file_paths = file_paths[size_val:]
		val = tf.data.Dataset.from_generator(loadImagesGen, args=(val_file_paths, label_names, h, w, resize), output_signature=(tf.TensorSpec(shape=(h, w, 3), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32))).batch(batch_size)
	
	if label_names==None:
		label_names = getLabelsNames(directory)
	return tf.data.Dataset.from_generator(loadImagesGen, args=(file_paths, label_names, h, w, resize), output_signature=(tf.TensorSpec(shape=(h, w, 3), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32))).batch(batch_size), val

