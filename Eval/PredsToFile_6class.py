# Given a model:
#   Write predictions to file: y_hat, y
#   Print basic metrics

from DataGen import AugSequence_v1 as as_v1
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, t
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def PredsToFile(model, datasrc="visible", filename="D:\Visible_code\Eval\PredsFile.csv"):
    crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    target_size = 256
    datasrc = "6class"

    dataGen = as_v1.AugSequence(crop_range=crop_range, allow_hor_flip=False, target_size=target_size, batch_size=32,
                                subtractMean=0.0, preprocess="div255", shuffle=False,
                                train_val_test="test", datasrc=datasrc, debug=False)

    # preds = model.predict_generator(dataGen, steps=len(dataGen))
    # labels = dataGen.dataGen().classes
    #preds_and_labels = pd.DataFrame()

    pred_scores = pd.DataFrame()
    labels_hot = pd.DataFrame()

    # Write predictions and labels to file
    for X, y in dataGen:
        pred_scores_batch = model.predict_on_batch(X)

        # Take Visible as TRUE class: 0th column in Invisible class; 1st - Visible
        #preds_and_labels_batch = np.stack([preds_batch[:, 1], y[:, 1]], axis=1)

        #preds_and_labels = preds_and_labels.append(pd.DataFrame(preds_and_labels_batch))
        pred_scores = pred_scores.append ( pd.DataFrame(pred_scores_batch) )
        labels_hot = labels_hot.append ( pd.DataFrame(y) )

    #preds_and_labels.to_csv(filename, header=None, index=None)

    #vis_preds = preds_and_labels.iloc[np.where(preds_and_labels.iloc[:, 1] >= 0.5)[0], 0]
    #invis_preds = preds_and_labels.iloc[np.where(preds_and_labels.iloc[:, 1] < 0.5)[0], 0]


    # Print out basic metrics
    preds = np.argmax (pred_scores.to_numpy(),  axis=1)
    labels = np.argmax (labels_hot.to_numpy(), axis=1)

    print ("Accuracy: {:7.4f}, Precision: {:7.4f}, Recall:  {:7.4f}, F1: {:7.4f}".format(
        accuracy_score(labels,preds),
        precision_score(labels, preds,average='macro'),
        recall_score(labels, preds,average='macro'),
        f1_score(labels, preds,average='macro'),
    ) )

    return (preds, labels)
