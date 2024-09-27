import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os

df_ent = pd.read_csv(os.environ['GDRIVE'] + "\\PhD_Data\\ClassMixture_Metrics\\eval_metrics.csv")

df_ent.Mean_Weighted

df_ent['Model (Vis/Invis)']

# x/y - number of visible/invisible
x = [ len(row.split("/")[0].split(",")) for row in df_ent['Model (Vis/Invis)'] ]
y = [ len(row.split("/")[1].split(",")) for row in df_ent['Model (Vis/Invis)'] ]

# add jitter so that circles don't fully overlap in range +-0.3
x += np.random.rand( len(x) ) * 0.5 - 0.25
y += np.random.rand( len(y) ) * 0.5 - 0.25


# doubling the width of markers
#x = np.arange(df_ent.shape[0])*2 #[0,2,4,6,8,10]
#y = np.random.randn( len(x) )
min_ent, mean_ent, max_ent = np.min(df_ent.Mean_Weighted), np.mean(df_ent.Mean_Weighted), np.max(df_ent.Mean_Weighted)
# scale so that size(ent_max) = 16**2 * size(ent_min); size(ent_min)=20
s = 20 + ( (df_ent.Mean_Weighted-min_ent) / (max_ent-min_ent) )**4 * (10**2-1)*20
#s = df_ent.Mean_Weighted**2*1000 #[20*4**n for n in range(len(x))]
#colors = np.zeros( (len(x) ,3) )
#colors [ np.where (s<2000)[0] ] = (1,0,0) # red for small circles
plt.scatter(x,y, s=s, alpha=0.3) # c=colors)

# #classes - tick marks
plt.xticks( np.arange(5)+1, labels=np.arange(5)+1)
plt.xticks( np.arange(5)+1, labels=np.arange(5)+1)

# separating lines
[ plt.axhline(i+1.5, linestyle='--', color='grey',linewidth=1) for i in np.arange(5) ]
[ plt.axvline(i+1.5, linestyle='--', color='grey',linewidth=1) for i in np.arange(5) ]

# axis labels and title
plt.xlabel ("Number of labels, Visible")
plt.ylabel ("Number of labels, Invisible")
plt.title ("Entropy = f (number of data labels in category)")

[s_min, s_mean, s_max] = np.min(s), np.mean(s), np.max(s)

legend_items = [ Line2D(range(1), range(1), color="white", marker='o',markersize=np.sqrt(s_item), markerfacecolor="slategray", alpha=0.5)
                 for s_item in [s_min, s_mean, s_max] ]
legend_texts = [ "{:.3f} (min)".format(min_ent), "{:.3f} (mean)".format(mean_ent),"{:.3f} (max)".format(max_ent) ]

plt.legend(
    legend_items,
    legend_texts,
    #title="Entropy",
    #title_fontsize=15,
    loc="upper right",
    labelspacing=2,     # vertical spacing between items
    #scatteryoffsets=[0.0375, 0.05, 0.03125],
    framealpha=0.5,
    borderpad = 3,      # increase size of legend box
    borderaxespad =0,   #move legend
    handletextpad=2)    # horizontal space between marker and text

plt.text (3.8,5.2, "Entropy", weight='bold',fontsize=15)

plt.grid(False)
plt.show()

