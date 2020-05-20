# Given a model:
#   Write predictions to file: y_hat, y
#   Print basic metrics

from DataGen import AugSequence_v1 as as_v1
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, t
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def PredsToFile(model, datasrc="visible", filename="D:\Visible_code\Eval\PredsFile.csv",
                vis_preds_filename="D:/Visible_code/Eval/vis_preds.csv",
                invis_preds_filename="D:/Visible_code/Eval/invis_preds.csv"):
    crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    target_size = 256
    datasrc = "6class"

    dataGen = as_v1.AugSequence(crop_range=crop_range, allow_hor_flip=False, target_size=target_size, batch_size=32,
                                subtractMean=0.0, preprocess="div255", shuffle=False,
                                train_val_test="test", datasrc=datasrc, debug=False)

    # preds = model.predict_generator(dataGen, steps=len(dataGen))
    # labels = dataGen.dataGen().classes
    preds_and_labels = pd.DataFrame()

    # Write predictions and labels to file
    for X, y in dataGen:
        preds_batch = model.predict_on_batch(X)

        # Take Visible as TRUE class: 0th column in Invisible class; 1st - Visible
        preds_and_labels_batch = np.stack([preds_batch[:, 1], y[:, 1]], axis=1)

        # 6 class
        #preds_and_labels_batch = np.stack([preds_batch, y], axis=1)

        preds_and_labels = preds_and_labels.append(pd.DataFrame(preds_and_labels_batch))

    preds_and_labels.to_csv(filename, header=None, index=None)

    vis_preds = preds_and_labels.iloc[np.where(preds_and_labels.iloc[:, 1] >= 0.5)[0], 0]
    invis_preds = preds_and_labels.iloc[np.where(preds_and_labels.iloc[:, 1] < 0.5)[0], 0]


    # do a t-test
    t_stat, p_value = ttest_ind(vis_preds, invis_preds)
    print ( "(ttest_ind) T-statistic: {:10.4f}, 1-tail p-value: {:E}".format(t_stat, p_value/2) )

    # Manually calculate t-stat and p-value (sanity check)
    vis_mean, invis_mean = np.mean(vis_preds), np.mean(invis_preds)
    vis_std, invis_std = np.std(vis_preds), np.std(invis_preds)
    vis_df, invis_df = len(vis_preds), len(invis_preds)
    t_stat_calculated = (vis_mean-invis_mean) / np.sqrt( vis_std*vis_std/vis_df+ + invis_std*invis_std/invis_df )
    p_value_calcuated = t.sf(np.abs(t_stat_calculated), vis_df+invis_df) # should df be sum of both samples? not sure
    # t.sf(np.abs(2.403), 50) == 1%
    print ( "(Manual) T-statistic: {:10.4f}, 1-tail p-value: {:E}".format(t_stat_calculated, p_value_calcuated) )

    # For sanity check, output preds to csv to check manually
    pd.DataFrame(vis_preds).to_csv( vis_preds_filename, header=None, index=None)
    pd.DataFrame(invis_preds).to_csv( invis_preds_filename, header=None, index=None)

    # Print out basic metrics
    pred_labels = np.round(preds_and_labels.iloc[:,0])
    true_labels = np.round(preds_and_labels.iloc[:,1])

    print ("Accuracy: {:7.4f}, Precision: {:7.4f}, Recall:  {:7.4f}, F1: {:7.4f}".format(
        accuracy_score(true_labels,pred_labels),
        precision_score(true_labels, pred_labels),
        recall_score(true_labels, pred_labels),
        f1_score(true_labels, pred_labels),
    ) )

    return preds_and_labels
