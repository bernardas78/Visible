from Model import Model_v1 as m_v1
from tensorflow.keras.callbacks import EarlyStopping
from numpy.random import seed
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


def trainModel(epochs=1, train_d1=False, train_d2=False,
               kfold_csv="10_filenames.csv", fold_id=-1, invisible_subcategories=["1"]):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    # Returns:
    #   model: trained Keras model
    #
    # To call:
    #   model = Train_v1.trainModel(epochs=20)

    #crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    target_size = 224
    batch_size = 128

    # define train and validation sets
    #   Load [fold,subcategory,filepath]
    df_kfold_split = pd.read_csv(kfold_csv)
    df_kfold_split["visible"] = df_kfold_split.subcategory.isin (invisible_subcategories).replace({True:"Invisible",False:"Visible"})
    df_train = df_kfold_split [ df_kfold_split.fold!=fold_id ]
    df_val = df_kfold_split [ df_kfold_split.fold==fold_id ]

    trainDataGen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=10,
        width_shift_range=32,
        height_shift_range=32,
        # brightness_range=[0.,2.],
        zoom_range=0.1,
        horizontal_flip=True
    )
    trainFlow = trainDataGen.flow_from_dataframe(dataframe=df_train, x_col="filepath", y_col="visible",
                                                 class_mode="categorical", target_size=(target_size, target_size),
                                                 batch_size=batch_size, shuffle=True)

    valDataGen = ImageDataGenerator(rescale = 1./255 )
    valFlow = valDataGen.flow_from_dataframe(dataframe=df_val, x_col="filepath", y_col="visible",
                                             class_mode="categorical", target_size=(target_size, target_size),
                                             batch_size=batch_size, shuffle=True)

    # Create model
    model = m_v1.prepModel( input_shape=(target_size,target_size,3) )

    # prepare a validation data generator, used for early stopping
    # vldDataGen = dg_v1.prepDataGen( target_size=target_size, test=True, batch_size=128, datasrc=datasrc )

    callback_earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10, verbose=1, mode='max',
                                       restore_best_weights=True)
    # callback_earlystop = EarlyStopping ( monitor='val_acc', min_delta=0., patience=0, verbose=2, mode='auto', restore_best_weights=True )

    # full epoch is 12x12 = 144 passes over data: 1 times for each subframe
    # model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2 )
    model.fit_generator(trainFlow, steps_per_epoch=len(trainFlow), epochs=epochs, verbose=2,
                        validation_data=valFlow, validation_steps=len(valFlow), callbacks=[callback_earlystop])

    # print ("Evaluation on train set (1 frame)")
    # e_v2.eval(model, target_size=target_size,  datasrc=datasrc)
    #print("Evaluation on validation set (1 frame)")
    #e_v2.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)
    #print("Evaluation on validation set (5 frames)")
    #e_v3.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)
    #print("Evaluation on validation set (10 frames)")
    #e_v4.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)

    return model
