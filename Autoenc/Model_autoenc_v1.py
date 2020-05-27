# Prepares a simple model
#   Sample downloaded from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#
# To run:
#   model = m_v1.prepModel()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Flatten, Reshape
from tensorflow.keras import regularizers

def prepModel( **argv ):
#def prepModel( input_shape, bn_layers, dropout_layers, l2_layers, padding, dense_sizes ):
    input_shape = argv["input_shape"]
    bn_layers = argv["bn_layers"]
    #dropout_layers = argv["dropout_layers"]
    #l2_layers = argv["l2_layers"]
    #padding = argv["padding"]
    #dense_sizes = argv["dense_sizes"]
    #   bn_layers - list of indexes of Dense layers (-1 and down) and CNN layers (1 and up) where Batch Norm should be applied
    #   dropout_layers - list of indexes of Dense layers (-1 and down) where Dropout should be applied
    #   bn_layers - list of indexes of Dense layers (-1 and down) where L2 regularization should be applied
    #   padding - changed to "same" to keep 2^n feature map sizes
    #   dense_sizes - dictionary of dense layer sizes (cnt of neurons)

    print ("Model_autoenc_v1")
    model = Sequential()

    ## ENCODER
    # 1st - encoder
    model.add(Convolution2D(8, (3,3), input_shape=input_shape, padding="same"))
    if "c+1" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd - encoder
    model.add(Convolution2D(16, (3, 3), padding="same"))
    if "c+2" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3rd - encoder
    model.add(Convolution2D(32, (3, 3), padding="same"))
    if "c+3" in bn_layers:
        model.add (BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 4th - encoder
    model.add(Convolution2D(64, (3, 3), padding="same"))
    if "c+4" in bn_layers:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 5th - encoder
    model.add(Convolution2D(128, (3, 3), padding="same"))
    if "c+5" in bn_layers:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # trivial layers: after Flatten - get activations and
    model.add (Flatten())
    model.add (Reshape( (8,8,128) ))

    ## DECODER
    # -5th decoder
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, (3, 3), padding="same"))
    if "c+5" in bn_layers:
        model.add(BatchNormalization())
    model.add(Activation('relu'))


    # -4th decoder
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, (3, 3), padding="same"))
    if "c+4" in bn_layers:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

# -3rd decoder
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(32, (3, 3), padding="same"))
    if "c+3" in bn_layers:
        model.add(BatchNormalization())
    model.add(Activation('relu'))


    # -2nd decoder
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(16, (3, 3), padding="same"))
    if "c+2" in bn_layers:
        model.add(BatchNormalization())
    model.add(Activation('relu'))


    # -1st decoder
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, (3, 3), padding="same"))
    if "c+1" in bn_layers:
        model.add(BatchNormalization())
    model.add(Activation('sigmoid'))


    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model