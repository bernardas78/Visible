from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
import os

version = 40
model_file_name = r"J:\Visible_models\model_6classes_v" + str(version) + ".h5"
conf_mat_file_name = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Conf_Mat\\model_6classes_v" + str(version) + "_conf_mat.png"
conf_mat_no_diag_file_name = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Conf_Mat\\model_6classes_v" + str(version) + "_conf_mat_no_diag.png"

data_dir_6classes_test = r"C:\TrainAndVal_6classes\Test"
#data_dir_6classes_test = r"D:\Visible_Data\3.SplitTrainValTest\Test"
#data_dir_6classes_test = r"C:\TrainAndVal_6classes\Val"

# Load and evaluate model on validation set
model = load_model (model_file_name)
#_,preds,labels,metrics = Metrics.CalcMetrics(model=model, model_type="resnet", datasrc = "ilsvrc14_10classes", testTest=True)

#target_size = 224 #up until v<=26
target_size = 256
batch_size = 32

class_names = os.listdir(data_dir_6classes_test)

dataGen = ImageDataGenerator( rescale=1. / 255 )

test_iterator = dataGen.flow_from_directory(directory=data_dir_6classes_test, target_size=(target_size, target_size),
                                           batch_size=batch_size, shuffle=False, class_mode='categorical')

Y_pred = model.predict_generator(test_iterator, len(test_iterator))
y_pred = np.argmax(Y_pred, axis=1)

#conf_mat = confusion_matrix(val_iterator.classes, y_pred))
conf_mat = confusion_matrix(y_true=test_iterator.classes,y_pred=y_pred) #, labels=actual_class_names)
#_preds, labels,_ = Metrics.CalcMetrics(model=model, model_type=model_type, datasrc = "ilsvrc14_10classes")

ax = sns.heatmap( conf_mat, annot=True, fmt='g', cbar=False )
ax.set_xticklabels( class_names, horizontalalignment='right' )
ax.set_yticklabels( class_names, horizontalalignment='right' )
ax.set(
      xlabel="Predicted",
      ylabel="Actual")

plt.savefig (conf_mat_file_name)
plt.close()

# No diagonal confusion matrix
for i in range(conf_mat.shape[0]):
    conf_mat[i,i] = 0
ax = sns.heatmap( conf_mat, annot=True, fmt='g', cbar=False )    #/np.sum(conf_mat)
ax.set_xticklabels( class_names, horizontalalignment='right' )
ax.set_yticklabels( class_names, horizontalalignment='right' )
plt.savefig (conf_mat_no_diag_file_name)
plt.close()
