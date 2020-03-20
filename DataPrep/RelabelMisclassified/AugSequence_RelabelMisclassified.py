import numpy as np
from PIL import ImageFile, Image
import time
import math
from tensorflow.keras.utils import Sequence

subcategory_names = ["1","2","3","4","m","ma"]

class AugSequence(Sequence):

    def __init__(self, df_data, target_size=224, batch_size=32, debug=False):

        print ("__init__, AugSequence_RelabelMisclassified.py ")
        self.target_size = target_size
        self.batch_size = batch_size
        self.debug = debug
        self.df_data = df_data
        self.img_filenames_cnt = len(self.df_data)

        # keep track how many batches requested. Based on this counter, proper batch will be returned
        self.cnter = 0

    # Length of sequence is length of directory iterator for each crop variant
    def __len__(self):
        return math.ceil( self.img_filenames_cnt / self.batch_size)

    def __getitem__(self, idx):

        # get next batch of images
        start_ind = np.min([self.cnter * self.batch_size, self.img_filenames_cnt])
        end_ind = np.min([(self.cnter + 1) * self.batch_size, self.img_filenames_cnt])
        img_filesnames_batch = self.df_data[start_ind: end_ind].filepath

        # Create X and y (extra class for non-class)
        # X = np.zeros ( (self.target_size,self.target_size,3,0 ) )
        X = np.zeros((len(img_filesnames_batch), self.target_size, self.target_size, 3))

        # One-hot encode label
        y = np.zeros ( (len(img_filesnames_batch), len(subcategory_names) ))
        y[np.arange(len(img_filesnames_batch)), self.df_data[start_ind:end_ind].subcategory.to_numpy(dtype=int) ] = 1

        img_counter_in_batch = 0

        # Read image data
        for img_filename in img_filesnames_batch:

            if self.debug and img_counter_in_batch == 0:
                print ("Reading file name " + img_filename + " Batch {0} in {1}".format(self.cnter, len(self)))
            img = Image.open(img_filename)

            img_resized = img.resize((self.target_size, self.target_size))

            img_rgb = img_resized.convert('RGB')

            img_preprocessed = np.asarray(img_rgb) * 1/255.

            X[img_counter_in_batch, :, :, :] = img_preprocessed

            img_counter_in_batch += 1
        # print ("X.shape:", X.shape)
        if self.debug:
            print('Batch {0} in {1}'.format(self.cnter, len(self)))

        # update counter
        self.cnter = (self.cnter+1) % len(self)

        # preserve filenames for getMinibatchFilenames()
        self.img_filesnames_batch = img_filesnames_batch

        return X, y

    # Return filenames of the last minibatch
    def getMinibatchFilenames(self):
        return self.img_filesnames_batch

    def on_epoch_end(self):
        #if self.cnter >= len(self):
        #    self.cnter = 0
        if self.debug:
            print("AugSequence_RelabelMisclassified.py, End of epoch")

    def __del__(self):
        if self.debug:
            print("AugSequence_RelabelMisclassified.py, __del__")
