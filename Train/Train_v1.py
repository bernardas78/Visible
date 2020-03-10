from DataGen import AugSequence_v1 as as_v1
from Model import Model_v1 as m_v1
from tensorflow.keras.callbacks import EarlyStopping
from numpy.random import seed
import tensorflow as tf
import numpy as np

myseed=1

def trainModel(epochs=1, train_d1=False, train_d2=False, use_class_weight=False):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    # Returns:
    #   model: trained Keras model
    #
    # To call:
    #   model = Train_v1.trainModel(epochs=20)

    seed(1)
    tf.random.set_seed(1)

    crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    target_size = 224
    datasrc = "visible"

    # define train and validation sets
    trainDataGen = as_v1.AugSequence(crop_range=crop_range, allow_hor_flip=False, target_size=target_size, batch_size=32,
                                subtractMean=0.0, preprocess="div255",
                                test=False, shuffle=True, datasrc=datasrc, debug=False)

    # use class weights for unbalanced classes
    class_weight=None
    if use_class_weight:
        class_keys, class_counts = np.unique(trainDataGen.dataGen().classes, return_counts=True)
        class_weight = dict ( zip (class_keys, np.sum(class_counts)/class_counts ) )
        print ("Using class weights:", class_weight)

    vldDataGen = as_v1.AugSequence(crop_range=crop_range, allow_hor_flip=False, target_size=target_size, batch_size=32,
                                subtractMean=0.0, preprocess="div255",
                                test=True, shuffle=True, datasrc=datasrc, debug=False)

    # Crete model
    model = m_v1.prepModel( input_shape=(target_size,target_size,3) )

    # prepare a validation data generator, used for early stopping
    # vldDataGen = dg_v1.prepDataGen( target_size=target_size, test=True, batch_size=128, datasrc=datasrc )

    callback_earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10, verbose=1, mode='max',
                                       restore_best_weights=True)
    # callback_earlystop = EarlyStopping ( monitor='val_acc', min_delta=0., patience=0, verbose=2, mode='auto', restore_best_weights=True )

    # full epoch is 12x12 = 144 passes over data: 1 times for each subframe
    # model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2 )
    model.fit_generator(trainDataGen, steps_per_epoch=len(trainDataGen), epochs=epochs, verbose=2,
                        validation_data=vldDataGen, validation_steps=len(vldDataGen), callbacks=[callback_earlystop],
                        class_weight=class_weight)

    # print ("Evaluation on train set (1 frame)")
    # e_v2.eval(model, target_size=target_size,  datasrc=datasrc)
    #print("Evaluation on validation set (1 frame)")
    #e_v2.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)
    #print("Evaluation on validation set (5 frames)")
    #e_v3.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)
    #print("Evaluation on validation set (10 frames)")
    #e_v4.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)

    return model
