# Prepares a simple model
#   Sample downloaded from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#
# To run:
#   model = m_v1.prepModel()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import regularizers
import numpy as np

def prepModel( input_shape, bn_layers, dropout_layers, l2_layers, padding, dense_sizes,
               conv_layers_over_5, use_maxpool_after_conv_layers_after_5th):
    #   bn_layers - list of indexes of Dense layers (-1 and down) and CNN layers (1 and up) where Batch Norm should be applied
    #   dropout_layers - list of indexes of Dense layers (-1 and down) where Dropout should be applied
    #   bn_layers - list of indexes of Dense layers (-1 and down) where L2 regularization should be applied
    #   padding - changed to "same" to keep 2^n feature map sizes
    #   dense_sizes - dictionary of dense layer sizes (cnt of neurons)
    #   conv_layers_over_5 - number of convolutional layers after 5th
    #   use_maxpool_after_conv_layers_after_5th - list of boolean values whether to use maxpooling after 5th layer
    print ("Model_6classes_c5plus_d3")

    model = Sequential()

    kernel_regularizer = regularizers.l2(0.01),

    # 1st CNN
    model.add(Convolution2D(32, (3,3), input_shape=input_shape, padding=padding))
    if "c+1" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd CNN
    model.add(Convolution2D(64, (3,3), padding=padding))
    if "c+2" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3rd CNN
    model.add(Convolution2D(128, (3, 3), padding=padding))
    if "c+3" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 4th CNN
    model.add(Convolution2D(256, (3, 3), padding=padding))
    if "c+4" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 5th CNN
    model.add(Convolution2D(256, (3, 3), padding=padding))
    if "c+5" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 6th+ CNN
    for layer_ind_after_5th in np.arange ( conv_layers_over_5 ):
        model.add(Convolution2D(256, (3, 3), padding=padding))
        if "c+"+str(layer_ind_after_5th+6) in bn_layers:
            model.add (BatchNormalization())
        model.add(Activation('relu'))
        if use_maxpool_after_conv_layers_after_5th[layer_ind_after_5th]:
            model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # -3rd dense
    d_3_regularizer = regularizers.l2( l2_layers["d-3"] ) if "d-3" in l2_layers else None
    d_3_size = dense_sizes["d-3"]
    model.add(Dense(d_3_size, kernel_regularizer=d_3_regularizer, bias_regularizer=d_3_regularizer))
    if "d-3" in bn_layers:
        model.add (BatchNormalization())
    if "d-3" in dropout_layers:
        model.add (Dropout(rate=0.5))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    # -2nd dense
    #model.add(Dense(128, activation='relu'))
    d_2_regularizer = regularizers.l2( l2_layers["d-2"] ) if "d-2" in l2_layers else None
    d_2_size = dense_sizes["d-2"]
    model.add(Dense(d_2_size, kernel_regularizer=d_2_regularizer, bias_regularizer=d_2_regularizer))
    if "d-2" in bn_layers:
        model.add (BatchNormalization())
    if "d-2" in dropout_layers:
        model.add (Dropout(rate=0.5))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    # -1st dense
    d_1_size = dense_sizes["d-1"] if "d-1" in dense_sizes else 6
    model.add(Dense(d_1_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model