# Runs all designed experiments (experiments_class_mixture.csv)

import pandas as pd

# experiment configurations
experiments_filename = "experiments_class_mixture.csv"

subcategories = ['1', '2', '3', '4', 'm', 'ma']


df_exper = pd.read_csv(experiments_filename)

for i,row in df_exper.iterrows():
    #print ("i:{}, row:{}".format(i,row) )

    Invisible_subcats = [ subcat for subcat in subcategories if row[subcat] == 0 ]
    Visible_subcats = [ subcat for subcat in subcategories if row[subcat] == 1 ]
    print ("Invisible:{}, Visible:{}".format(Invisible_subcats,Visible_subcats) )
