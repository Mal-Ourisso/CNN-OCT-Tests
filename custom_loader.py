
import tensorflow as tf

from random import shuffle, seed
from glob import glob

def getFilePaths(directory):
        return [file_path for file_path in glob(directory+"/**/*.jpeg")]

def getLabels(directory):
        return [getLabelName(directory, label) for label in glob(directory+"/*/")]

def getLabelName(directory, file_path):
        directory = directory.split("/")
        file_path = file_path.split("/")
        label_path = list(filter(lambda x: x not in directory, file_path))
        return label_path[0]

def getImageWithLabel(directory, file_path, label_names, h=512, w=512, resize="interpolation"):
        file_path = str(file_path).split("'")[1]

        img = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)
        if resize=="interpolation":
                resize = lambda img, w, h: tf.image.resize(img, [w, h])
        elif resize=="zeropadding":
                resize = lambda img, w, h: tf.image.resize_with_pad(img, w, h)
        x = resize(img, w, h)

        label = getLabelName(directory, file_path)
        y = 0
        for name in label_names:
                if label == name:
                        break
                y+=1
        return x, y

def genImagesWithLabels(directory, file_paths, label_names=None, rand=True, h=512, w=512, resize="interpolation"):
        directory = str(directory).split("'")[1]
        resize = str(resize).split("'")[1]
        if not label_names:
                label_names = getLabels(directory)
        for path in file_paths:
                yield getImageWithLabel(directory, path, label_names, h, w, resize)

def customDataset(directory, rand=True, validation_split=0, label_names=False, batch_size=64, h=512, w=512, resize="interpolation"):
        file_paths = getFilePaths(directory)
        if rand:
                if isinstance(rand, int):
                        seed(rand)
                shuffle(file_paths)

        val = None
        if validation_split > 0:
                size_val = int(validation_split*len(file_paths))
                val_file_paths = file_paths[:size_val]
                file_paths = file_paths[size_val:]
                val = tf.data.Dataset.from_generator(
                        genImagesWithLabels,
                        args=(directory, val_file_paths, label_names, rand, h, w, resize),
                        output_signature=(tf.TensorSpec(shape=(h, w, 3), dtype=tf.int32),
                        tf.TensorSpec(shape=(), dtype=tf.int32))).batch(batch_size)


        dataset = tf.data.Dataset.from_generator(
                genImagesWithLabels,
                args=(directory, file_paths, label_names, rand, h, w, resize),
                output_signature=(tf.TensorSpec(shape=(h, w, 3), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32))).batch(batch_size)

        return dataset, val
