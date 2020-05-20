from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from matplotlib import pyplot as plt
import os

version = 40
model_file_name = r"J:\Visible_models\2class\model_2classes_v" + str(version) + ".h5"
roc_file_name = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\ROC\\model_2classes_v" + str(version) + "_roc.png"

data_dir_2classes_test = r"C:\TrainAndVal_2classes\Test"

# Load and evaluate model on validation set
model = load_model (model_file_name)
#_,preds,labels,metrics = Metrics.CalcMetrics(model=model, model_type="resnet", datasrc = "ilsvrc14_10classes", testTest=True)

#target_size = 224 #up until v<=26
target_size = 256
batch_size = 32

class_names = os.listdir(data_dir_2classes_test)

dataGen = ImageDataGenerator( rescale=1. / 255 )

test_iterator = dataGen.flow_from_directory(directory=data_dir_2classes_test, target_size=(target_size, target_size),
                                           batch_size=batch_size, shuffle=False, class_mode='categorical')

Y_pred_scores = model.predict_generator(test_iterator, len(test_iterator))

# Visible class is Positive class; Visible's index is 1
visible_index = class_names.index("Visible")
Visible_pred_scores = Y_pred_scores[:,visible_index]
Visible_labels = test_iterator.classes==visible_index

# ROC (hypothetical)
plt.figure(figsize=(8, 6))  # Not shown
fpr, tpr, _ = roc_curve(Visible_labels, Visible_pred_scores)
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
plt.axis([0, 1, 0, 1])  # Not shown in the book
plt.xlabel('1 - Specificity', fontsize=16)  # Not shown
plt.ylabel('Recall', fontsize=16)  # Not shown
plt.title ("ROC AUC={}".format(roc_auc_score(Visible_labels, Visible_pred_scores)))
plt.grid(True)  # Not shown

plt.savefig (roc_file_name)
plt.close()

