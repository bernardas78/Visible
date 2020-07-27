# Copies augmented data to Visible/Invisible (undersampled)

import shutil
import os

# Source folder with augmented 6-class images
src_data_folder = "D:\\Visible_Data\\4.Augmented"
# Data folder of 2-class images which will be recreated
dest_data_folder = "C:\\TrainAndVal"
# Backup folder to copy data to for later review
backup_folder = "J:\\ClassMixture_Data"


def create_dataset (Subcats, version):
    # Subcats: dictionary {"Visible: ["4",..], "Invisible: []}
    print ("create_dataset vis/invis {}/{}".format(Subcats["Visible"], Subcats["Invisible"]))

    # Copy data into Invisible/Visible folders
    recreate_folders = [ "Train", "Val", "Test" ]
    for the_folder in recreate_folders:

        the_full_folder = os.path.join(dest_data_folder, the_folder)

        # Recreate Train|Val folders
        if os.path.exists ( the_full_folder ):
            shutil.rmtree ( the_full_folder )
        os.mkdir ( the_full_folder )

        # Copy data to category folders
        for cat in Subcats: #
            os.mkdir ( os.path.join (the_full_folder, cat))

            cnt_visible_subcats = len(Subcats[cat])
            for subcat in Subcats[cat]:
                for i, img_filename in enumerate( os.listdir( os.path.join(src_data_folder, the_folder, subcat) ) ):
                    # under-sample if more than 1 subcat in cat (except test set)
                    if i%cnt_visible_subcats==0 or the_folder=="Test":
                        shutil.copyfile ( os.path.join (src_data_folder, the_folder, subcat, img_filename),
                                          os.path.join (dest_data_folder, the_folder, cat, img_filename))

            # Save file counts (for debuging)
            #filecounts = []

    # Create a data copy for later review
    exper_backup_folder = os.path.join(backup_folder, "TrainAndVal_experId_" + str(version))
    if os.path.exists(exper_backup_folder):
        shutil.rmtree(exper_backup_folder)
    shutil.copytree( dest_data_folder, exper_backup_folder )

