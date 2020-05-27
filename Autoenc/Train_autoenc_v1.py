from Autoenc import Model_autoenc_v1 as m_autoenc_v1
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#modelVersions_dic = {
#    "Model_6classes_c4_d3_v1": m_6classes_c4_d3_v1.prepModel,
#    "Model_6classes_c5_d2_v1": m_6classes_c5_d2_v1.prepModel,
#    "Model_6classes_c5_d3_v1": m_6classes_c5_d3_v1.prepModel,
#    "Model_6classes_c6_d2_v1": m_6classes_c6_d2_v1.prepModel,
#    "Model_6classes_c5plus_d3_v1": m_6classes_c5plus_d3_v1.prepModel
#}

def trainModel(epochs,bn_layers, target_size, version):

    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    #   bn_layers - list of indexes of Dense layers (-1 and down) and CNN layers (1 and up) where Batch Norm should be applied
    #   dropout_layers - list of indexes of Dense layers (-1 and down) where Dropout should be applied
    #   bn_layers - list of indexes of Dense layers (-1 and down) where L2 regularization should be applied
    #   padding - changed to "same" to keep 2^n feature map sizes
    #   dense_sizes - dictionary of dense layer sizes (cnt of neurons)
    #   architecture - one of:  Model_6classes_c4_d3_v1, Model_6classes_c5_d2_v1, Model_6classes_c5_d3_v1
    #   conv_layers_over_5 - number of convolutional layers after 5th
    #   use_maxpool_after_conv_layers_after_5th - list of boolean values whether to use maxpooling after 5th layer
    #   version - used to name a learning curve file
    # Returns:
    #   model: trained Keras model
    #
    # To call:
    #   model = Train_v1.trainModel(epochs=20)

    crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    #target_size = 224
    batch_size = 32
    #datasrc = "visible"

    # Manually copied to C: to speed up training
    data_dir_6classes_train = r"C:\TrainAndVal_6classes\Train"
    data_dir_6classes_val = r"C:\TrainAndVal_6classes\Val"
    data_dir_6classes_test = r"C:\TrainAndVal_6classes\Test"
    #data_dir_6classes_train = r"D:\Visible_Data\4.Augmented\Train"
    #data_dir_6classes_val = r"D:\Visible_Data\4.Augmented\Val"

    # define train and validation sets
    trainValDataGen = ImageDataGenerator(
        #rotation_range=5, #15, #10,
        #width_shift_range=16, #48, #32,
        #height_shift_range=16, #48, #32,
        # brightness_range=[0.,2.],
        #zoom_range=0.05, #0.15, #0.1,
        #horizontal_flip=True,
        rescale=1./255
    )
    train_iterator = trainValDataGen.flow_from_directory(
        directory=data_dir_6classes_train,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='input')

    val_iterator = trainValDataGen.flow_from_directory(
        directory=data_dir_6classes_val,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='input')

    #trainDataGen = s_6classes_v1.AugSequence(crop_range=crop_range, allow_hor_flip=False, target_size=target_size, batch_size=32,
    #                            subtractMean=0.0, preprocess="div255",
    #                            test=False, shuffle=True, datasrc=datasrc, debug=False)


    # Create model
    prepModel = m_autoenc_v1.prepModel  #modelVersions_dic[architecture]
    prep_model_params = {
        "input_shape": (target_size,target_size,3),
        "bn_layers": bn_layers,
        #"dropout_layers": dropout_layers,
        #"l2_layers": l2_layers,
        #"padding": padding,
        #"dense_sizes": dense_sizes,
        #"conv_layers_over_5": conv_layers_over_5,
        #"use_maxpool_after_conv_layers_after_5th": use_maxpool_after_conv_layers_after_5th
    }
    model = prepModel (**prep_model_params)
    #model = m_6classes_c4_d3_v1.prepModel( input_shape=(target_size,target_size,3),
    #                                       bn_layers=bn_layers, dropout_layers=dropout_layers, l2_layers=l2_layers,
    #                                       padding=padding, dense_sizes=dense_sizes )
    print (model.summary())

    # prepare a validation data generator, used for early stopping
    # vldDataGen = dg_v1.prepDataGen( target_size=target_size, test=True, batch_size=128, datasrc=datasrc )

    callback_earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=20, verbose=1, mode='max',
                                       restore_best_weights=True)
    callback_csv_logger = CSVLogger('J:/Visible_models/LearningCurves/lc_model_autoenc_v' + str(version) + '.csv', separator=",", append=False)


    # full epoch is 12x12 = 144 passes over data: 1 times for each subframe
    # model.fit_generator ( dataGen, steps_per_epoch=len(dataGen), epochs=epochs, verbose=2 )

    model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=epochs, verbose=2,
                        validation_data=val_iterator, validation_steps=len(val_iterator), callbacks=[callback_earlystop,callback_csv_logger])

    # End experiments with dataset size

    #print ("Evaluating on test set")
    #testDataGen = ImageDataGenerator(
    #    rescale=1./255
    #)
    #test_iterator = testDataGen.flow_from_directory(
    #    directory=data_dir_6classes_test,
    #    target_size=(target_size, target_size),
    #    batch_size=batch_size,
    #    shuffle=True,
    #    class_mode='categorical')
    #test_loss = model.evaluate_generator ( test_iterator, steps=len(test_iterator) )
    #print ("Test loss: {0}, accuracy: {1}".format(test_loss[0], test_loss[1]))

    return model
