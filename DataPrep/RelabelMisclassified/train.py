from DataPrep.RelabelMisclassified import  splitTrainValStratified as split
from DataPrep.RelabelMisclassified import Model_6classes_RelabelMisclassified as m_6classes_rm
from DataPrep.RelabelMisclassified import AugSequence_RelabelMisclassified as as_6classes_rm
from tensorflow.keras.callbacks import EarlyStopping

def trainModel(epochs=1, bn_layers = [], dropout_layers=[], l2_layers={}):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    #   bn_layers - list of indexes of Dense layers (-1 and down) and CNN layers (1 and up) where Batch Norm should be applied
    #   dropout_layers - list of indexes of Dense layers (-1 and down) where Dropout should be applied
    #   bn_layers - list of indexes of Dense layers (-1 and down) where L2 regularization should be applied
    # Returns:
    #   model: trained Keras model
    #
    # To call:
    #   model = Train_v1.trainModel(epochs=20)

    crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    target_size = 224
    batch_size = 64
    #datasrc = "visible"

    # define train and validation sets
    df_train, df_val = split.splitTrainVal()
    train_iterator = as_6classes_rm.AugSequence (df_data=df_train, target_size=target_size, batch_size=batch_size, debug=False)
    val_iterator = as_6classes_rm.AugSequence (df_data=df_val, target_size=target_size, batch_size=batch_size, debug=False)

    # Create model
    model = m_6classes_rm.prepModel( input_shape=(target_size,target_size,3), bn_layers=bn_layers, dropout_layers=dropout_layers, l2_layers=l2_layers )

    callback_earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=20, verbose=1, mode='max',
                                       restore_best_weights=True)

    # full epoch is 12x12 = 144 passes over data: 1 times for each subframe
    # model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2 )
    model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=epochs, verbose=2,
                        validation_data=val_iterator, validation_steps=len(val_iterator), callbacks=[callback_earlystop])

    return model
