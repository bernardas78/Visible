# Creates SOM-based classifiers and evaluates them (using several SOM hyper-parameters:
#   1. Create SOM clusters on train
#   2. Assign most frequest class to a cluster
#   3. Evaluate classifier on train, val, test
#
# Src: Orange table files (created by pickle_to_orange.py)
# Dest: accuracies.csv

import pandas as pd
import Orange
import os
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
from Orange.projection import som
import time
import numpy as np
import pickle

subcategories = ["1","2","3","4","m","ma"]

# Orange table files
orange_encoded_activations_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Autoenc"
orange_train_activations_filename = "\\".join ([orange_encoded_activations_folder,"train_activations_enc.tab"])
orange_val_activations_filename = "\\".join ([orange_encoded_activations_folder,"val_activations_enc.tab"])
orange_test_activations_filename = "\\".join ([orange_encoded_activations_folder,"test_activations_enc.tab"])

# Load orange table files
now = time.time()
train_tab = Orange.data.Table.from_file( filename=orange_train_activations_filename )
val_tab = Orange.data.Table.from_file( filename=orange_val_activations_filename )
test_tab = Orange.data.Table.from_file( filename=orange_test_activations_filename )
print ("Loaded orange data files in {} seconds".format (time.time()-now) )

# Dest:
df_accs_filename = os.path.join( orange_encoded_activations_folder, "accuracies.csv" )

def callback(iter):
    print ("Callback called after iter {}".format(iter))
    return True

# Calculate train accuracy
def get_accuracy_som(dim_x, dim_y, n_iterations, l_rate):
    print ("Training SOM on ({},{}) grid for {} iterations, learning rate {}".format(dim_x, dim_y, n_iterations, l_rate) )
    # Dimension of SOM grid
    mysom = som.SOM(dim_x=dim_x, dim_y=dim_y)

    # Train SOM on training set
    now = time.time()

    mysom.fit( x=train_tab.X, n_iterations=n_iterations, learning_rate=l_rate, callback=callback)

    print ("Trained SOM for {} seconds".format (time.time()-now) )
    now = time.time()

    #   Get winner neurons for train, val, test sets
    pred_train_winner_neurons = mysom.winners ( train_tab.X )
    pred_val_winner_neurons = mysom.winners ( val_tab.X )
    pred_test_winner_neurons = mysom.winners ( test_tab.X )

    #   Get winner class for each neuron - from train set
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
        if class_winners_in_neurons [winner_x, winner_y] == train_tab.Y[sample_id]:
            train_tp +=1
    train_acc = float(train_tp) /  train_tab.X.shape[0]

    # Calculate validation accuracy
    val_tp = 0
    for sample_id in np.arange(val_tab.X.shape[0]):
        #   Determine class of sample
        winner_x, winner_y = pred_val_winner_neurons[sample_id, :]
        if class_winners_in_neurons [winner_x, winner_y] == val_tab.Y[sample_id]:
            val_tp +=1
    val_acc = float(val_tp) /  val_tab.X.shape[0]

    # Calculate test accuracy
    test_tp = 0
    for sample_id in np.arange(test_tab.X.shape[0]):
        #   Determine class of sample
        winner_x, winner_y = pred_test_winner_neurons[sample_id, :]
        if class_winners_in_neurons [winner_x, winner_y] == test_tab.Y[sample_id]:
            test_tp +=1
    test_acc = float(test_tp) /  test_tab.X.shape[0]

    # Log metrics
    df_accs = pd.DataFrame(
        data=[np.hstack([dim_x, dim_y, n_iterations, l_rate, train_acc, val_acc, test_acc])],
        columns=column_names)
    df_accs.to_csv(df_accs_filename, header=None, index=None, mode='a')

    print ("Accuracy train: {}, val: {}, test: {}. Eval took {} secs".format (train_acc, val_acc, test_acc, time.time()-now))
    return (train_acc, val_acc, test_acc, mysom)

##################################
# Run a grid search for SOM
##################################
column_names = ['dim_x','dim_y','n_iterations','learning_rate','train_acc','val_acc','test_acc']
df_accs = pd.DataFrame (columns=column_names )
# Uncomment to overwrite
df_accs.to_csv(df_accs_filename, index=False, header=True, mode='w')

dims = [8, 16, 32]
n_iterations = [1, 5, 10, 20, 50]
l_rates = [0.5]

dims = [32]
n_iterations = [180]
l_rates = [0.75]


for dim in dims:
    for l_rate in l_rates:
        for n_iteration in n_iterations:
            dim_x, dim_y = dim,dim
            (train_acc, val_acc, test_acc, mysom) = get_accuracy_som(dim_x=dim, dim_y=dim, n_iterations=n_iteration, l_rate=l_rate)

            # save som
            filehandler = open("J:\\Visible_models\\SOM\\som_v2.h5", 'wb')
            pickle.dump(mysom, filehandler)
            filehandler.close()