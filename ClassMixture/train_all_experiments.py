# Runs all designed experiments (experiments_class_mixture.csv - which is created by list_class_mixture.py)

import pandas as pd
from train_one_experiment import trainSingleClassMixtureExperiment

# Starting point in case it crashed
start_model = 24
end_model = 24

# experiment configurations
experiments_filename = "experiments_class_mixture.csv"

subcategories = ['1', '2', '3', '4', 'm', 'ma']


df_exper = pd.read_csv(experiments_filename)

for i,row in df_exper.iterrows():
    #print ("i:{}, row:{}".format(i,row) )

    if i<start_model:
        continue
    if i>end_model:
        continue

    Invisible_subcats = [ subcat for subcat in subcategories if row[subcat] == 0 ]
    Visible_subcats = [ subcat for subcat in subcategories if row[subcat] == 1 ]
    Subcats = {"Visible": Visible_subcats, "Invisible": Invisible_subcats}
    #print ("Invisible:{}, Visible:{}".format(Invisible_subcats,Visible_subcats) )

    trainSingleClassMixtureExperiment (Subcats=Subcats, version=i)

    # for debugging; remove
    #if i>=1:
    #    break
