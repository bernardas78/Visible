# Draw learning curves (acc) for variant data set sizes
# Draw acc = f (dataset size)

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

ds_size_versions = [3647,4014,4395,4823]

test_accuracies = {
    4823: 0.6256366968154907, #v1: 0.5738539695739746,
    4395: 0.5993208885192871,
    4014: 0.5797962546348572,
    3647: 0.5636672377586365
}

best_val_accuracies = {}

colors = {
    4823: "red",
    4395: "green",
    4014: "blue",
    3647: "black"
}

lc_file_template = "J:\\Visible_models\\DataseSizeExperiments\\lc_model_6classes_v56_"

lc_fig_filename = "J:\\Visible_models\\DataseSizeExperiments\\lc_variantDatasetSizes.jpg"
acc_dssize_fig_filename = "J:\\Visible_models\\DataseSizeExperiments\\acc_variantDatasetSizes.jpg"

max_epochs=0

for ds_size in ds_size_versions:

    # Load LC data file
    lc_data_filename = lc_file_template + str(ds_size) + ".csv"
    metrics = pd.read_csv (lc_data_filename)

    max_epochs = max(max_epochs, np.max(metrics["epoch"]))

    #model_filename = model_folder + "model_6classes_v" + str(version) + ".h5"
    #print ("Loading model {}...".format(model_filename))
    #model = load_model (model_filename)
    #print ("Model loaded")

    # Evaluate on statically-augmented set and not-augmented set
    #print ("Evaluating on static-only augmentation")
    #dataGen = ImageDataGenerator( rescale=1./255 )
    #static_aug_iterator = dataGen.flow_from_directory(
    #    directory=data_dir_static_aug[version],
    #    target_size=(256, 256),
    #    batch_size=32,
    #    class_mode='categorical')
    #static_aug_metrics = model.evaluate_generator ( static_aug_iterator, steps=len(static_aug_iterator) )
    #static_aug_accuracies[version] = static_aug_metrics[1]
    #print ("Static-only loss: {0}, accuracy: {1}".format(static_aug_metrics[0], static_aug_metrics[1]))

    #print ("Evaluating on no augmentation")
    #dataGen = ImageDataGenerator( rescale=1./255 )
    #no_aug_iterator = dataGen.flow_from_directory(
    #    directory=data_dir_no_aug,
    #    target_size=(256, 256),
    #    batch_size=32,
    #    class_mode='categorical')
    #no_aug_metrics = model.evaluate_generator ( no_aug_iterator, steps=len(no_aug_iterator) )
    #no_aug_accuracies[version] = no_aug_metrics[1]
    #print ("No-augmentation loss: {0}, accuracy: {1}".format(no_aug_metrics[0], no_aug_metrics[1]))

    # PLOT1: learning curves
    # Val accuracy learning curves
    line_val, = plt.plot( metrics["epoch"], metrics["val_accuracy"], color=colors[ds_size])
    line_val.set_label( "Val, " + str(ds_size) )

    # Determine point of early stopping (where best val accuracy was);
    best_val_acc_epoch = np.argmax(metrics["val_accuracy"])

    # Preserve best val accuracy for later plot
    best_val_accuracies[ds_size] = metrics["val_accuracy"][best_val_acc_epoch]

    # Draw points for test acc and best-val-acc point
    pt_test, = plt.plot(best_val_acc_epoch, test_accuracies[ds_size], marker="o", color=colors[ds_size])
    pt_test.set_label( "Test, " + str(ds_size) )

# PLOT 1: learning curves for variant dataset sizes
plt.legend(title='')
plt.xlabel("Epoch")
plt.ylabel("Val, Test Accuracy")
x = np.arange(max_epochs)
plt.xticks(ticks=x,labels=x+1, rotation=90)
plt.title ( "Variant Dataset sizes, Accuracy=f(epoch)" )
plt.savefig(lc_fig_filename)
plt.close()



# PLOT2: Accuracy=f(ds_size)
plt.title ("Accuracy = f (Dataset Size)")
plt.xlabel("Train+Val set size")
plt.ylabel("Val, Test Accuracy")

# Validation acc
line_val, = plt.plot( ds_size_versions, [best_val_accuracies[ds_size] for ds_size in ds_size_versions], color="red")
line_val.set_label("Validation Acc")

# Test acc
line_val, = plt.plot( ds_size_versions, [test_accuracies[ds_size] for ds_size in ds_size_versions], color="blue")
line_val.set_label("Test Acc")

# X ticks - train+val set size
plt.xticks (ticks=ds_size_versions, labels=ds_size_versions, rotation=90)

plt.legend(title='')
plt.tight_layout()  #to avoid clipping rotated xticks
plt.savefig(acc_dssize_fig_filename)
#plt.show()
plt.close()
