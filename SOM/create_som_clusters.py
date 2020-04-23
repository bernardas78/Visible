# Create SOM clusters
#      on pre-last layer's activations of trained NN classifier
#           activations written to file by ErrorAnalysis\mispred_activations_preLast_to_file_for_knn.py

# Docs on Orange:
#   Create table: https://docs.biolab.si//3/data-mining-library/reference/data.table.html
#                 https://docs.biolab.si//3/data-mining-library/reference/data.domain.html#Orange.data.Domain
#   Fit SOM:
#       D:\Programs\Anaconda3\Lib\site-packages\Orange\projection\som.py

import Orange
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
from Orange.projection import som
import pickle
import os
import numpy as np
import time
import pandas as pd

subcategories = ["1","2","3","4","m","ma"]

mispred_knn_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Misclsf_Knn"
som_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\SOM"
#   Pre-last layer activations
train_activations_filename = "\\".join ([mispred_knn_folder,"train_activations_preLast.obj"])
val_activations_filename = "\\".join ([mispred_knn_folder,"val_activations_preLast.obj"])
test_activations_filename = "\\".join ([mispred_knn_folder,"test_activations_preLast.obj"])

# Results file name
df_accs_filename =  "\\".join ([som_folder,"accuracies.csv"])

# Load activation files
train_result = pickle.load( open(train_activations_filename, 'rb') )
val_result = pickle.load( open(val_activations_filename, 'rb') )
test_result = pickle.load( open(test_activations_filename, 'rb') )
print ("Loaded activations files")

# unpack train activations, file names, classes
(train_filenames, train_classes, train_pred_scores, train_activations_preLast) = train_result
(val_filenames, val_classes, val_pred_scores, val_activations_preLast) = val_result
(test_filenames, test_classes, test_pred_scores, test_activations_preLast) = test_result
print ("Unpacked activations: ")
print ("  train_filenames {0}, train_classes {1}, train_activations_preLast {2} ".format(len(train_filenames), len(train_classes), train_activations_preLast.shape))
print ("  val_filenames {0}, val_classes {1}, val_pred_scores {2}, val_activations_preLast {3} ".format(len(val_filenames), len(val_classes), val_pred_scores.shape, val_activations_preLast.shape))
print ("  test_filenames {0}, test_classes {1}, test_pred_scores {2}, test_activations_preLast {3} ".format(len(test_filenames), len(test_classes), test_pred_scores.shape, test_activations_preLast.shape))

# convert to Orange table and save
#   Later used for:
#       a) Visual Orange SOM
#       b) Here - later in code - SOM clustering
domain = Domain(
            [ContinuousVariable.make("Feat_"+str(i)) for i in np.arange(train_activations_preLast.shape[1])],
            DiscreteVariable.make(name="subcategory", values=subcategories) )
train_class_indices = np.asarray ( [subcategories.index(train_class) for train_class in train_classes] )
train_tab = Orange.data.Table.from_numpy( domain=domain, X=train_activations_preLast, Y=train_class_indices)

val_class_indices = np.asarray ( [subcategories.index(val_class) for val_class in val_classes] )
val_tab = Orange.data.Table.from_numpy( domain=domain, X=val_activations_preLast, Y=val_class_indices)

# Save Orange table files
train_activations_orangeTable_filename = "\\".join ([mispred_knn_folder,"train_activations_preLast.tab"])
train_tab.save (train_activations_orangeTable_filename)

val_activations_orangeTable_filename = "\\".join ([mispred_knn_folder,"val_activations_preLast.tab"])
val_tab.save (val_activations_orangeTable_filename)

print ("Saved Orange files (train, val)")
#train_tab = Orange.data.Table.from_file(train_activations_orangeTable_filename)

#iris_tab = Orange.data.Table.from_file("D:\\Programs\\Orange\\lib\\site-packages\\Orange\\datasets\\iris.tab")
#iris1_tab = Orange.data.Table.from_numpy( domain=iris_tab.domain, X=iris_tab.X, Y=iris_tab.Y )

# Calculate train accuracy
def get_accuracy_som(dim_x, dim_y, n_iterations, l_rate):
    print ("Training SOM on ({},{}) grid for {} iterations, learning rate {}".format(dim_x, dim_y, n_iterations, l_rate) )
    # Dimension of SOM grid
    mysom = som.SOM(dim_x=dim_x, dim_y=dim_y)

    now = time.time()
    mysom.fit( x=train_tab.X, n_iterations=n_iterations, learning_rate=l_rate )
    print ("Trained SOM for {} seconds".format (time.time()-now) )

    # Evaluate train error
    #   Get winner neurons for train set
    pred_train_winner_neurons = mysom.winners ( train_tab.X )
    pred_val_winner_neurons = mysom.winners ( val_tab.X )


    #   Get winner class for each neuron
    class_counts_in_neurons = np.zeros ( (dim_x, dim_y, len(subcategories)), dtype='int')
    #       First, count each class instances in neurons
    for sample_id in np.arange(train_tab.X.shape[0]):
        # increase class count for winner neuron
        winner_x,winner_y = pred_train_winner_neurons [sample_id,:]
        class_ind = int(train_tab.Y [sample_id])
        class_counts_in_neurons [winner_x,winner_y,class_ind] +=1
    #       Then, assign highest class
    class_winners_in_neurons = np.zeros ( (dim_x, dim_y), dtype='int')
    for row in np.arange(dim_y):
        for col in np.arange(dim_x):
            class_winners_in_neurons[row,col] = np.argsort( class_counts_in_neurons[row,col,:] ) [-1]

    # Calculate training accuracy
    train_tp = 0
    for sample_id in np.arange(train_tab.X.shape[0]):
        #   Determine class of sample
        winner_x, winner_y = pred_train_winner_neurons[sample_id, :]
        if class_winners_in_neurons [winner_x, winner_y] == train_class_indices[sample_id]:
            train_tp +=1
    train_acc = float(train_tp) /  train_tab.X.shape[0]

    # Calculate validation accuracy
    val_tp = 0
    for sample_id in np.arange(val_tab.X.shape[0]):
        #   Determine class of sample
        winner_x, winner_y = pred_val_winner_neurons[sample_id, :]
        if class_winners_in_neurons [winner_x, winner_y] == val_class_indices[sample_id]:
            val_tp +=1
    val_acc = float(val_tp) /  val_tab.X.shape[0]

    print ("Accuracy train: {}, val: {}".format (train_acc, val_acc))
    return (train_acc, val_acc)

##################################
# Run a grid search for SOM
##################################
column_names = ['dim_x','dim_y','n_iterations','learning_rate','train_acc','val_acc']
df_accs = pd.DataFrame (columns=column_names )
# Uncomment to overwrite
#df_accs.to_csv(df_accs_filename, index=False, header=True, mode='w')

dims = [8, 16, 21, 32]
n_iters = [10,50,100,200]
l_rates = [0.25, 0.5, 1]

for dim in dims:
    for n_iter in n_iters:
        for l_rate in l_rates:

            dim_x, dim_y = dim,dim
            (train_acc, val_acc) = get_accuracy_som(dim_x=dim, dim_y=dim, n_iterations=n_iter, l_rate=l_rate)

            df_accs = pd.DataFrame(
                data=[np.hstack([dim_x, dim_y, n_iter, l_rate, train_acc, val_acc])],
                columns=column_names)
            df_accs.to_csv(df_accs_filename, header=None, index=None, mode='a')