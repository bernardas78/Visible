from Model_6class import Model_6classes_v1 as m_6classes_v1
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def trainModel(epochs=1, train_d1=False, train_d2=False, use_class_weight=False, bn_layers = [], dropout_layers=[], l2_layers={}):
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

    # Manually copied to C: to speed up training
    data_dir_6classes_train = r"C:\TrainAndVal_6classes\Train"
    data_dir_6classes_val = r"C:\TrainAndVal_6classes\Val"
    #data_dir_6classes_train = r"D:\Visible_Data\4.Augmented\Train"
    #data_dir_6classes_val = r"D:\Visible_Data\4.Augmented\Val"

    # define train and validation sets
    dataGen = ImageDataGenerator(
        rotation_range=5, #10,
        width_shift_range=16, #32,
        height_shift_range=16, #32,
        # brightness_range=[0.,2.],
        zoom_range=0.05, #0.1,
        horizontal_flip=False, #True,
        rescale=1./255
    )
    train_iterator = dataGen.flow_from_directory(
        directory=data_dir_6classes_train,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    val_iterator = dataGen.flow_from_directory(
        directory=data_dir_6classes_val,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')
    #trainDataGen = s_6classes_v1.AugSequence(crop_range=crop_range, allow_hor_flip=False, target_size=target_size, batch_size=32,
    #                            subtractMean=0.0, preprocess="div255",
    #                            test=False, shuffle=True, datasrc=datasrc, debug=False)


    # Crete model
    model = m_6classes_v1.prepModel( input_shape=(target_size,target_size,3), bn_layers=bn_layers, dropout_layers=dropout_layers, l2_layers=l2_layers )
    print (model.summary())

    # prepare a validation data generator, used for early stopping
    # vldDataGen = dg_v1.prepDataGen( target_size=target_size, test=True, batch_size=128, datasrc=datasrc )

    callback_earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=20, verbose=1, mode='max',
                                       restore_best_weights=True)
    # callback_earlystop = EarlyStopping ( monitor='val_acc', min_delta=0., patience=0, verbose=2, mode='auto', restore_best_weights=True )

    # full epoch is 12x12 = 144 passes over data: 1 times for each subframe
    # model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2 )
    model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=epochs, verbose=2,
                        validation_data=val_iterator, validation_steps=len(val_iterator), callbacks=[callback_earlystop])

    # print ("Evaluation on train set (1 frame)")
    # e_v2.eval(model, target_size=target_size,  datasrc=datasrc)
    #print("Evaluation on validation set (1 frame)")
    #e_v2.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)
    #print("Evaluation on validation set (5 frames)")
    #e_v3.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)
    #print("Evaluation on validation set (10 frames)")
    #e_v4.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)

    return model
