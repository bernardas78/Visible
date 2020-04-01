# Plot distribution of probabilities using results of 100 clsf
#   Input: Result of 100 classifiers: D:\Google Drive\PhD_Data\Visible_ErrorAnalysis\Relabelling\Preds100Clsf.csv
#   Outs: plot in ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\100clsf_analysis\\

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

subcategory_names = ["1","2","3","4","m","ma"]

filename_100clsf_res = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Relabelling\\Preds100Clsf.csv"

plots_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\100clsf_analysis\\"

do_distribution_probabilities_of_actual_class = True
do_distribution_probabilities_of_actual_vs_other_classes = True
do_distribution_probabilities_of_each_class_when_actual_class_is_x = True


print ("Loading Preds100Clsf.csv...")
df_100clsf_res = pd.read_csv ( filename_100clsf_res, header=0)
print ("Loaded Preds100Clsf.csv")

# Overall disribution of probabilities
if do_distribution_probabilities_of_actual_class:
    f, axarr = plt.subplots(3, 2)
    f.suptitle (" Overall distribution of probabilities ")
    for idx,ax in enumerate(axarr.ravel()):
        subcategory_name = subcategory_names[idx]
        ax.set_title(subcategory_name, pad=-15)
        _ = ax.hist(df_100clsf_res[subcategory_name], bins=50)
    plt.savefig(plots_folder + "overall_distribution_probabilities.jpg")
    print ("saved overall_distribution_probabilities.jpg")
    plt.show()


# Disribution of probabilities when actual=class; actual!=class;
if do_distribution_probabilities_of_actual_vs_other_classes:
    f, axarr = plt.subplots(3, 2)
    f.suptitle ("Distribution of probabilities of class X:\n when actual=X (BLUE) and actual<>X (RED)")
    for idx,ax in enumerate(axarr.ravel()):
        subcategory_name = subcategory_names[idx]

        df_probs_of_actual = df_100clsf_res.loc [df_100clsf_res.actual==idx][subcategory_name]
        df_probs_of_others = df_100clsf_res.loc [df_100clsf_res.actual!=idx][subcategory_name]

        ax.set_title(subcategory_name, pad=-15)
        _ = ax.hist(df_probs_of_actual, bins=20, color="blue", alpha = 0.5)
        _ = ax.hist(df_probs_of_others, bins=20, color="red", alpha = 0.5)
    plt.savefig(plots_folder + "distribution_probabilities_of_actual_vs_other_classes.jpg")
    print ("saved distribution_probabilities_of_actual_vs_other_classes.jpg")
    plt.show()

# How are probabilities of each class distributed when actual class is x
if do_distribution_probabilities_of_each_class_when_actual_class_is_x:
    f, axarr = plt.subplots(3, 2, figsize=(16,8))
    f.suptitle ("Distribution of probabilities of classes\n when actual=X")
    #f.set_size_inches(6, 6)

    num_bins = 10
    x = np.arange(num_bins) * 0.1
    width = 1./num_bins / len(subcategory_names) -0.002 # subtract some to interleave between probs

    for true_subcategory_id,ax in enumerate(axarr.ravel()):
        true_subcategory_name = subcategory_names[true_subcategory_id]

        # when actual==x
        df_actual = df_100clsf_res.loc [df_100clsf_res.actual==true_subcategory_id]

        # a rectangle for each subcategory
        for pred_subcategory_id,pred_subcategory_name in enumerate(subcategory_names):
            probs_predicted = df_actual[pred_subcategory_name]
            pred_class_counts = np.histogram(probs_predicted, bins=10)[0]
            #print ("Bfr:", len(x), pred_subcategory_id)
            #print(pred_class_counts)
            #print(x + pred_subcategory_id*width )
            _ = ax.bar(x + pred_subcategory_id*width, pred_class_counts, width, label=pred_subcategory_name)

        ax.set_title("Actual="+true_subcategory_name, pad=-15)
        if true_subcategory_id == 1:
            ax.legend()

    plt.savefig(plots_folder + "distribution_probabilities_of_each_class_when_actual_is_x.jpg")
    print ("saved distribution_probabilities_of_each_class_when_actual_is_x.jpg")
    plt.show()
