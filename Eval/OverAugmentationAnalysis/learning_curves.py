# Draw learning curves (acc); add points for static-only agmented data and not augmented data

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

versions = [40,43,44,45,46,47,48]

model_folder = "J:\\Visible_models\\"

data_dir_no_aug = "D:\\Visible_Data\\3.SplitTrainValTest\\Train"

data_dir_static_aug = {
    40: "D:\\Visible_Data\\4.Augmented_FullAug\\Train",     # incl. dynamic aug
    43: "D:\\Visible_Data\\4.Augmented_FullAug\\Train",     # no dynamic aug
    44: "D:\\Visible_Data\\4.Augmented_HalfAug\\Train",     # half static
    45: "D:\\Visible_Data\\4.Augmented_MinusRot\\Train",    # static - rotation
    46: "D:\\Visible_Data\\4.Augmented_MinusShift\\Train",  # static - shift
    47: "D:\\Visible_Data\\4.Augmented_MinusZoom\\Train",   # static - zoom
    48: "D:\\Visible_Data\\4.Augmented_MinusHorFlip\\Train" # static - hor flip
}

model_names = {
    40: "Dynamic + Static",
    43: "Static only",  # no dynamic aug
    44: "Half static",  # half static
    45: "Static minus\n rotation",  # static - rotation
    46: "Static minus\n shift",  # static - shift
    47: "Static minus\n zoom",  # static - zoom
    48: "Static minus\n horizontal flip"  # static - hor flip
}

# Initialize to collect each model's train accuracy on static-aug set
static_aug_accuracies = {}
# Initialize to collect each model's train accuracy on no-aug set
no_aug_accuracies = {}
# Initialize to collect each model's train accuracy at the last training epoch
last_epoch_accuracies = {}
# Initialize to collect each model's train accuracy at the reverted-to-best-val-acc point
reverted_accuracies = {}

for version in versions:

    # Load LC data file
    lc_data_filename = model_folder + "LearningCurves\\lc_model_6classes_v" + str(version) + ".csv"
    metrics = pd.read_csv (lc_data_filename)

    model_filename = model_folder + "model_6classes_v" + str(version) + ".h5"
    print ("Loading model {}...".format(model_filename))
    model = load_model (model_filename)
    print ("Model loaded")

    # Evaluate on statically-augmented set and not-augmented set
    print ("Evaluating on static-only augmentation")
    dataGen = ImageDataGenerator( rescale=1./255 )
    static_aug_iterator = dataGen.flow_from_directory(
        directory=data_dir_static_aug[version],
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')
    static_aug_metrics = model.evaluate_generator ( static_aug_iterator, steps=len(static_aug_iterator) )
    static_aug_accuracies[version] = static_aug_metrics[1]
    print ("Static-only loss: {0}, accuracy: {1}".format(static_aug_metrics[0], static_aug_metrics[1]))

    print ("Evaluating on no augmentation")
    dataGen = ImageDataGenerator( rescale=1./255 )
    no_aug_iterator = dataGen.flow_from_directory(
        directory=data_dir_no_aug,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')
    no_aug_metrics = model.evaluate_generator ( no_aug_iterator, steps=len(no_aug_iterator) )
    no_aug_accuracies[version] = no_aug_metrics[1]
    print ("No-augmentation loss: {0}, accuracy: {1}".format(no_aug_metrics[0], no_aug_metrics[1]))

    # PLOT1: learning curves
    # Train and accuracy learning curves
    line_train, = plt.plot( metrics["epoch"], metrics["accuracy"], color="blue")
    line_val, = plt.plot( metrics["epoch"], metrics["val_accuracy"], color="red")

    line_train.set_label("Train")
    line_val.set_label("Validation")

    # Determine point of early stopping (where best val accuracy was);
    best_val_acc_epoch = np.argmax(metrics["val_accuracy"])
    # Train Accuracy at best-val-acc point
    reverted_accuracies[version] = metrics.loc[best_val_acc_epoch].accuracy

    # Last Train Accuracy
    last_epoch_accuracies[version] = metrics.loc[ metrics.shape[0]-1 ].accuracy

    # Draw points for train-static-only-aug and train-no-aug
    pt_st_aug, = plt.plot(best_val_acc_epoch, static_aug_metrics[1], marker="o", color="green")
    pt_st_aug.set_label("Static-only augmentation")
    pt_no_aug, = plt.plot(best_val_acc_epoch, no_aug_metrics[1], marker="o", color="brown")
    pt_no_aug.set_label("No augmentation")


    plt.legend(title='')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(ticks=metrics["epoch"],labels=metrics["epoch"]+1, rotation=90)
    plt.title ( model_names[version] + ", Accuracy=f(epoch)" )

    #plt.show()
    lc_fig_filename = model_folder + "LearningCurves\\lc_model_6classes_v" + str(version) + ".jpg"
    plt.savefig(lc_fig_filename)
    plt.close()



# PLOT2: bar plot of all models' train accuracies
x = np.arange ( len(versions) )
width = 1. / 5

_ = plt.bar(x + 0*width, last_epoch_accuracies.values(), width, label="Last training epoch")
_ = plt.bar(x + 1*width, reverted_accuracies.values(), width, label="Reverted to best-val point")
_ = plt.bar(x + 2*width, static_aug_accuracies.values(), width, label="Static only augmentation")
_ = plt.bar(x + 3*width, no_aug_accuracies.values(), width, label="No augmentation")

plt.title ("Train accuracy for variant augmentations")
plt.legend(title='Train accuracy', loc='lower left')
plt.xticks (ticks=x, labels=[model_names[i] for i in versions], rotation=45)
plt.tight_layout()  #to avoid clipping rotated xticks

bars_fig_filename = model_folder + "LearningCurves\\variant_augmentations.jpg"

plt.savefig(bars_fig_filename)
plt.show()

plt.close()
