from Model_6class import Model_6classes_c4_d3_v1 as m_6classes_c4_d3_v1
from Model_6class import Model_6classes_c5_d2_v1 as m_6classes_c5_d2_v1
from Model_6class import Model_6classes_c5_d3_v1 as m_6classes_c5_d3_v1
from Model_6class import Model_6classes_c6_d2_v1 as m_6classes_c6_d2_v1
from ClassMixture import Model_ClassMixture_v1 as m_classmixture_v1
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import os

# Calc and save test accuracies
test_accuracies_filename = os.environ['GDRIVE'] + "\\PhD_Data\\ClassMixture_Metrics\\test_loss_accuracies.csv"
# ...and other metrics
test_metrics_filename = os.environ['GDRIVE'] + "\\PhD_Data\\ClassMixture_Metrics\\test_metrics.csv"
# conf mat file pattern
conf_mat_file_pattern = os.environ['GDRIVE'] + "\\PhD_Data\\ClassMixture_Metrics\\Conf_Mat\\conf_mat_v{}.png"

modelVersions_dic = {
    "Model_classmixture_v1": m_classmixture_v1.prepModel
}

def trainModel(epochs,bn_layers, dropout_layers, l2_layers,
               padding, target_size, dense_sizes,
               architecture, conv_layers_over_5, use_maxpool_after_conv_layers_after_5th, version, load_existing):

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
    #   load_existing - whether to load an existing model file
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
    data_dir_train = r"C:\TrainAndVal\Train"
    data_dir_val = r"C:\TrainAndVal\Val"
    data_dir_test = r"C:\TrainAndVal\Test"

    # Save Learning curve
    lc_filepath_pattern = 'J:/ClassMixture_LearningCurves/lc_v'

    # define train and validation sets
    trainValDataGen = ImageDataGenerator(
        rotation_range=10, #5, #15, #10,             #5-single, 15-triple, 10-double dynamic augmentation
        width_shift_range=32, #16, #48, #32,
        height_shift_range=32, #16, #48, #32,
        # brightness_range=[0.,2.],
        zoom_range=0.1, #0.05, #0.15, #0.1,
        horizontal_flip=True,
        rescale=1./255
    )
    train_iterator = trainValDataGen.flow_from_directory(
        directory=data_dir_train,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    val_iterator = trainValDataGen.flow_from_directory(
        directory=data_dir_val,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    #trainDataGen = s_6classes_v1.AugSequence(crop_range=crop_range, allow_hor_flip=False, target_size=target_size, batch_size=32,
    #                            subtractMean=0.0, preprocess="div255",
    #                            test=False, shuffle=True, datasrc=datasrc, debug=False)


    # Create model
    if not load_existing:
        print ("Creating model")
        prepModel = modelVersions_dic[architecture]
        prep_model_params = {
            "input_shape": (target_size,target_size,3),
            "bn_layers": bn_layers,
            "dropout_layers": dropout_layers,
            "l2_layers": l2_layers,
            "padding": padding,
            "dense_sizes": dense_sizes,
            "conv_layers_over_5": conv_layers_over_5,
            "use_maxpool_after_conv_layers_after_5th": use_maxpool_after_conv_layers_after_5th
        }
        model = prepModel (**prep_model_params)
    else:
        print ("Loading model")
        model_file_name = r"J:\ClassMixture_Models\model_v" + str(version) + ".h5"
        model = load_model(model_file_name)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.001), # default LR: 0.001
                      metrics=['accuracy'])

    #model = m_6classes_c4_d3_v1.prepModel( input_shape=(target_size,target_size,3),
    #                                       bn_layers=bn_layers, dropout_layers=dropout_layers, l2_layers=l2_layers,
    #                                       padding=padding, dense_sizes=dense_sizes )
    print (model.summary())

    # prepare a validation data generator, used for early stopping
    # vldDataGen = dg_v1.prepDataGen( target_size=target_size, test=True, batch_size=128, datasrc=datasrc )

    if not load_existing:
        callback_earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=20, verbose=1, mode='max',
                                           restore_best_weights=True)
        callback_csv_logger = CSVLogger(lc_filepath_pattern + str(version) + '.csv', separator=",", append=False)

        model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=epochs, verbose=2,
                            validation_data=val_iterator, validation_steps=len(val_iterator), callbacks=[callback_earlystop,callback_csv_logger])

    print ("Evaluating on test set")

    # Overwrite metric files?
    mode = 'a' if version>0 else 'w'
    header = version==0

    testDataGen = ImageDataGenerator( rescale=1./255 )
    test_iterator = testDataGen.flow_from_directory(
        directory=data_dir_test,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')
    #test_loss, test_acc = model.evaluate_generator ( test_iterator, steps=len(test_iterator) )
    #print ("Test loss: {0}, accuracy: {1}".format(test_loss, test_acc))
    #df_accuracies = pd.DataFrame(columns=["version","test_loss","test_acc"],
    #                             data=[np.hstack([version, test_loss, test_acc])])
    #df_accuracies.to_csv(test_accuracies_filename, header=header, index=None, mode=mode)

    # calc other important metrics: precision, recall, f1 (also acc for sanity check)
    predictions = model.predict_generator(test_iterator, steps=len(test_iterator))
    pred_classes = np.argmax(predictions, axis=1)

    acc = accuracy_score(test_iterator.classes, pred_classes),
    prec = precision_score(test_iterator.classes, pred_classes),
    recall = recall_score(test_iterator.classes, pred_classes),
    f1 = f1_score(test_iterator.classes, pred_classes)
    df_metrics = pd.DataFrame(columns=["version","acc","prec","recall","f1"],
                                 data=[np.hstack([version, acc, prec, recall, f1 ])])
    df_metrics.to_csv(test_metrics_filename, header=header, index=None, mode=mode)
    print("Test accuracy: {}, precision: {}, recall: {}, f1: {}".format(acc, prec, recall, f1))

    # conf matrix
    conf_mat = confusion_matrix(y_true=(1 - test_iterator.classes), y_pred=(1 - pred_classes))

    sns.set(font_scale=3.0)
    ax = sns.heatmap(np.round(conf_mat / np.sum(conf_mat) * 100, decimals=1), annot=True, fmt='.1f', cbar=False)
    for t in ax.texts: t.set_text(t.get_text() + " %")
    ax.set_xticklabels(list(reversed(list(test_iterator.class_indices))), horizontalalignment='right', size=20)  # switch category names, like we switched data
    ax.set_yticklabels(list(reversed(list(test_iterator.class_indices))), horizontalalignment='right', size=20)
    ax.set_xlabel("PREDICTED", weight="bold", size=20)
    ax.set_ylabel("ACTUAL", weight="bold", size=20)
    plt.tight_layout()
    plt.savefig(conf_mat_file_pattern.format(version))
    plt.close()

    # print ("Evaluation on train set (1 frame)")
    # e_v2.eval(model, target_size=target_size,  datasrc=datasrc)
    #print("Evaluation on validation set (1 frame)")
    #e_v2.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)
    #print("Evaluation on validation set (5 frames)")
    #e_v3.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)
    #print("Evaluation on validation set (10 frames)")
    #e_v4.eval(model, target_size=target_size, datasrc=datasrc, preprocess="vgg", test=True)

    return model
