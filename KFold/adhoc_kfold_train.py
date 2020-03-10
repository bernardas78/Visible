from KFold import Train_kfold_v1 as t_kfold_v1

cv_folds = 5

start_fold = 0

invisible_subcategories=["1"]

for fold_id in range(start_fold, cv_folds):
    model_file_name = "J:\\Visible_models\\KFold\\model_2classes_fold_" + str(fold_id) + "_of_" + str(cv_folds) + "_v1.h5"
    kfold_csv = str(cv_folds) + "_filenames.csv"

    model = t_kfold_v1.trainModel(epochs=100, kfold_csv=kfold_csv, fold_id=fold_id, invisible_subcategories=invisible_subcategories)
    model.save (model_file_name)

