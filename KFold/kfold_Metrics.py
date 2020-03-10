import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def CalcMetrics(model, kfold_csv="10_filenames.csv", fold_id=-1, invisible_subcategories=["1"]):

    crop_range = 1 # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    target_size = 224
    batch_size = 128

    # define train and validation sets
    #   Load [fold,subcategory,filepath]
    df_kfold_split = pd.read_csv(kfold_csv)
    df_kfold_split["visible"] = df_kfold_split.subcategory.isin (invisible_subcategories).replace({True:"Invisible",False:"Visible"})
    df_train = df_kfold_split [ df_kfold_split.fold!=fold_id ]
    df_val = df_kfold_split [ df_kfold_split.fold==fold_id ]

    valDataGen = ImageDataGenerator(rescale = 1./255 )
    valFlow = valDataGen.flow_from_dataframe(dataframe=df_val, x_col="filepath", y_col="visible",
                                             class_mode="categorical", target_size=(target_size, target_size),
                                             batch_size=batch_size, shuffle=False)

    # Get predictions and labels on entire validation dataset
    #preds_and_labels = pd.DataFrame()

    # Initialize empty array for predictions and labels
    #pred_scores = np.empty( (0,10), dtype='int' )
    #preds = np.empty( (0), dtype='int' )
    #labels = np.empty( (0), dtype='int' )

    visible_class_index = valFlow.class_indices["Visible"]
    pred_scores_visible = model.predict_generator( generator=valFlow, steps=len(valFlow) )[:,visible_class_index]
    preds = np.round(pred_scores_visible)
    labels = valFlow.classes
    #for X, y in dataGen:
        #preds_batch = model.predict_on_batch(X)

        # Take Visible as TRUE class: 0th column in Invisible class; 1st - Visible
        #preds_and_labels_batch = np.stack([  np.argmax(preds_batch, axis=1)  , np.argmax(y, axis=1) ], axis=1)

        #preds_and_labels = preds_and_labels.append(pd.DataFrame(preds_and_labels_batch))
        #preds_and_labels = np.concatenate ( (preds_and_labels, preds_and_labels_batch), axis=0 )

        #pred_scores = np.concatenate ( (pred_scores, preds_batch ) )
        #preds = np.concatenate ( (preds, np.argmax(preds_batch, axis=1) ) )
        #labels = np.concatenate ( (labels, np.argmax(y, axis=1) ) )

    metrics = {"accuracy":accuracy_score(preds, labels),
               "precision":precision_score(preds, labels, average='macro'),
               "recall":recall_score(preds, labels, average='macro'),
               "f1":f1_score(preds, labels, average='macro')}
    print("Accuracy: ", metrics["accuracy"])
    print("Precision: ", metrics["precision"]) #macro - calc metric for each label and average them (unweighted)
    print("Recall: ", metrics["recall"])
    print("F1: ", metrics["f1"])

    return metrics