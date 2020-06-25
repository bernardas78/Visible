# Lists all experiments for class mixture into  experiments_class_mixture.csv

import numpy as np
import pandas as pd

# Combinatorial grouping of all sub-classes
experiments = []
subcategories = ['1', '2', '3', '4', 'm', 'ma']
#subcat_data = [1923, 1329, 912, 1242, 438, 157]

# where to save experiment configurations
experiments_filename = "experiments_class_mixture.csv"

for experiment_id in range(3 * 3 * 3 * 3):
    # print('experiment_id={}'.format(experiment_id))

    # 0th class - invisible; 1st - visible; 2nd - not included
    subcats_class_in_experiment = [
        0,  # subcat '1' always in 0th class
        int(experiment_id / 1) % 3,  # subcat '2'
        int(experiment_id / 3) % 3,  # subcat '3'
        1,  # subcat '4' always in 1st class
        int(experiment_id / (3 * 3)) % 3,  # subcat 'm'
        int(experiment_id / (3 * 3 * 3)) % 3]  # subcat 'ma'

    # exclude illogical experiments
    # ['3','1'] vs ['2','4']
    # ['ma','4'] vs ['m','1']
    if (subcats_class_in_experiment[2] == 0 and subcats_class_in_experiment[1] == 1) \
            or (subcats_class_in_experiment[4] == 0 and subcats_class_in_experiment[5] == 1):
        print ("Skipping:{}".format(subcats_class_in_experiment) )
        continue

    experiments.append(subcats_class_in_experiment)


# column names: classes (what each class consists of)
column_names = subcategories
df_exper = pd.DataFrame(columns=column_names, data=np.array(experiments))
df_exper.to_csv(experiments_filename, index=False, header=True, mode='w')

#for dic_key in experiments:
#    df_exper = pd.DataFrame(
#        # data=[np.hstack(experiments[dic_key])],
#        data=[np.hstack(
#            [str([subcat_ind for subcat_ind in range(6) if experiments[dic_key][subcat_ind] == theclass]) for theclass
#             in range(6)])],
#        columns=column_names)
#    df_exper.to_csv(df_exper_classes_filename, header=None, index=None, mode='a')

# print ("exp_id:{}, classes:{}".format(dic_key, experiments[dic_key]))
