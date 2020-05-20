import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import time
#from keras.applications.vgg16 import preprocess_input
from numpy.random import seed
import tensorflow as tf

myseed = 1

class AugSequence(keras.utils.Sequence):

    seed(myseed)
    tf.random.set_seed(myseed)

    def __init__(self, crop_range=1, allow_hor_flip=False, target_size=224, batch_size=32, \
                 subtractMean=0.0, preprocess="div255", \
                 train_val_test="train", shuffle=True, datasrc="visible", debug=False):

        self.target_size = target_size
        self.crop_range = crop_range
        self.allow_hor_flip = allow_hor_flip
        self.subtractMean = subtractMean
        self.debug = debug

        # it used to throw file truncated error. bellow makes it tolerant to truncated files
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if datasrc == "visible":
            if train_val_test=="train":
                data_dir = "C:/TrainAndVal_Visible/train"
            elif train_val_test == "val":
                data_dir = "C:/TrainAndVal_Visible/val"
            else:
                data_dir = "C:/TrainAndVal_Visible/test"
        elif datasrc == "6class":
            if train_val_test == "train":
                data_dir = "C:/TrainAndVal_6classes/train"
            if train_val_test == "val":
                data_dir = "C:/TrainAndVal_6classes/val"
            else:
                data_dir = "C:/TrainAndVal_6classes/test"
        else:
            raise Exception('AugSequence_v1: unknown datasrc')

        if preprocess == "div255":
            datagen = ImageDataGenerator(rescale=1. / 255)
        else:
            raise Exception('AugSequence_v1: unknown preprocess parameter')

        size_uncropped = target_size + crop_range - 1

        self.data_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(size_uncropped, size_uncropped),
            batch_size=batch_size,
            shuffle=shuffle,
            class_mode='categorical',
            seed=myseed)

        # store length for faster retrieval of length
        self.len_value = len(self.data_generator)

        # keep track how many items requested. Based on this counter, proper crop to be returned
        self.cnter = 0

        # initiate async thread for augmented data retrieval, which will be received via __getitem__()

    # Length of sequence is length of directory iterator for each crop variant
    def __len__(self):
        return self.len_value

    def __getitem__(self, idx):
        # print ( "Starting getitem" )

        # get next uncropped batch of images
        X_uncropped, y = next(self.data_generator)

        # get proper crop based on counter
        start_w = np.random.randint(0, self.crop_range)
        start_h = np.random.randint(0, self.crop_range)
        horflip = np.random.choice(a=[False, True])
        if self.allow_hor_flip and horflip:
            X = np.flip(X_uncropped[:, start_w:start_w + self.target_size, start_h:start_h + self.target_size, :],
                        axis=1)
        else:
            X = X_uncropped[:, start_w:start_w + self.target_size, start_h:start_h + self.target_size, :]

        # subtract to center values
        X -= self.subtractMean

        # update counter : max value is len of entire imageset
        self.cnter += 1

        if self.debug and self.cnter % 100 == 0:
            print("AugSequence_v1.py, __getitem__, self.cnter, self.len_value:", str(self.cnter), " ", self.len_value, " ",
                  time.strftime("%H:%M:%S"))
        # print ("X.shape, y.shape, start_w, start_h, target_size, self.cnter", X.shape, y.shape, start_w, start_h, self.target_size, self.cnter)

        return X, y

    def on_epoch_end(self):
        self.data_generator.reset()
        if self.debug:
            print("AugSequence_v1.py, on_epoch_end")

    def __del__(self):
        if self.debug:
            print("AugSequence_v1.py, __del__")

    def dataGen(self):
        return self.data_generator
