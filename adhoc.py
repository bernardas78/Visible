from Train import Train_v1 as t_v1
from Eval import PredsToFile as ptf
from numpy.random import seed
import tensorflow as tf

#Repro
seed(1)
tf.random.set_seed(1)

model = t_v1.trainModel(epochs=100, use_class_weight=True)

preds_and_labels = ptf.PredsToFile(model)