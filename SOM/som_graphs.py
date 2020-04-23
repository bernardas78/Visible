# Create charts for SOM results
#   Src: accuracies.csv (obtained by running create_som_clusters.py)
#   Dest: accuracies.png

import os
import pandas as pd
#import matplotlib.markers as mmark
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
import numpy as np

som_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\SOM"
df_accs_filename =  "\\".join ([som_folder,"accuracies.csv"])
df_graph_filename =  "\\".join ([som_folder,"accuracies.png"])

df_som_accs = pd.read_csv ( df_accs_filename, header=0)


#
#sizes_lst = [ 1, 4, 8, 16 ]
#sizes = [ np.unique(df_som_accs.dim_x).tolist().index (dim_x) for dim_x in df_som_accs.dim_x ]

dim_x_values = (np.unique(df_som_accs.dim_x)).tolist()

colors_set = [ "red", "green", "blue", "orange" ]
learning_rate_values = np.unique(df_som_accs.learning_rate).tolist()

markers_set = [ "v", "<", "^", ">" ]
n_iterations_values = np.unique(df_som_accs.n_iterations).tolist()

fig, ax = plt.subplots()

for i, n_iterations_value in enumerate(n_iterations_values):
    df_som_accs_single_marker = df_som_accs.loc [df_som_accs.n_iterations==n_iterations_value]
    sizes = df_som_accs_single_marker.dim_x * df_som_accs_single_marker.dim_x / 4
    colors = [colors_set[learning_rate_values.index(learning_rate)] for learning_rate in df_som_accs_single_marker.learning_rate]
    scatter = ax.scatter (x=df_som_accs_single_marker.train_acc, y=df_som_accs_single_marker.val_acc, s=sizes, c=df_som_accs_single_marker.learning_rate, marker=markers_set [i] )


#legend1 = ax.legend(*scatter.legend_elements(num=5),
#                    loc="upper left", title="Ranking")
#ax.add_artist(legend1)

# Produce a legend for the price (sizes). Because we want to show the prices
# in dollars, we use the *func* argument to supply the inverse of the function
# used to calculate the sizes from above. The *fmt* ensures to show the price
# in dollars. Note how we target at 5 elements here, but obtain only 4 in the
# created legend due to the automatic round prices that are chosen for us.

kw = dict(prop="colors")
legend = plt.legend(*scatter.legend_elements(**kw), loc="lower left", title="Lerning rate")
ax.add_artist(legend)

kw = dict(prop="sizes", func=lambda s: np.sqrt(s*4))
legend2 = plt.legend(*scatter.legend_elements(**kw), loc="lower right", title="SOM Grid size")
ax.add_artist(legend2)

#kw = dict(prop="markers")
#legend3 = plt.legend(*scatter.legend_elements(**kw), loc="lower middle", title="Iterations")
#ax.legend( [mmark.MarkerStyle(marker) for marker in markers_set] , markers_set)
markers_legend_handles = []
for i,marker in enumerate(markers_set):
    markers_legend_handles.append (
        mlines.Line2D([], [], marker=marker, label=int(n_iterations_values[i])))
ax.legend( handles=markers_legend_handles, loc="lower center", title="Iterations")

plt.xlabel ("Train accuracy")
plt.ylabel ("Validation accuracy")
plt.title ("SOM clustering=>Classification")

plt.show()
plt.savefig(df_graph_filename)