import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np

def twoclass_conf_mat_to_file (y_true, y_pred, conf_mat_filename):
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)

    sns.set(font_scale=3.0)
    ax = sns.heatmap(np.round(conf_mat / np.sum(conf_mat) * 100, decimals=1), annot=True, fmt='.1f', cbar=False)
    for t in ax.texts: t.set_text(t.get_text() + " %")
    ax.set_xticklabels(["Invisible","Visible"], horizontalalignment='right', size=20)
    ax.set_yticklabels(["Invisible","Visible"], horizontalalignment='right', size=20)
    ax.set_xlabel("PREDICTED", weight="bold", size=20)
    ax.set_ylabel("ACTUAL", weight="bold", size=20)
    plt.tight_layout()
    plt.savefig( conf_mat_filename )
    plt.close()