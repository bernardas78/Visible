import os
import numpy as np
from matplotlib import pyplot as plt

subcat_counts_chart_filename = os.environ['GDRIVE'] + "\\PhD_Data\\ClassMixture_Metrics\\subcat_counts_chart.png"
data_folder = r'D:\Visible_Data\2.Cropped'

subcat_counts = {}

translated_class_names = {"1": "Q1",
                         "2": "Q2",
                         "3": "Q3",
                         "4": "Q4",
                         "m": "Bag",
                         "ma": "BagR"}
for subcat_folder in os.listdir(data_folder):
    subcat_counts[subcat_folder] = len(os.listdir( os.path.join(data_folder,subcat_folder) ) )

print (subcat_counts)

plt.barh ( y=np.arange(len(subcat_counts)), width=subcat_counts.values(), tick_label=[translated_class_names[key] for key in subcat_counts.keys()] )
plt.xlabel ("Number of labelled images")
plt.ylabel ("Class name")
plt.title ("Image counts per class")
plt.savefig (subcat_counts_chart_filename)
plt.show()