from Eval.PredsToFile_6class import PredsToFile
from tensorflow.keras.models import load_model

model = load_model ("J:\\Visible_models\\model_6classes_v40.h5")

preds_and_labels = PredsToFile (model=model, datasrc="6class")