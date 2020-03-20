from DataPrep.RelabelMisclassified import trainAndLogPreds as tlp
import pandas as pd
import numpy as np
import gc
import os

subcategory_names = ["1","2","3","4","m","ma"]

predsFile = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Relabelling\\Preds100Clsf.csv"

#bn_layers = ["c+1", "c+2", "c+3", "c+4", "d-2", "d-3"]  # v19


df_results =  pd.DataFrame ( columns= ["model", "filename","actual","predicted"] + subcategory_names )
#df_results.to_csv(predsFile, header=True, index=None, mode='w')

for model_id in np.arange(90,100):
    model = tlp.trainAndLogPreds(predsFile=predsFile, epochs=100, model_id_to_log=model_id) #, bn_layers=bn_layers

    del model
    gc.collect()

