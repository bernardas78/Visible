# Splits to train/test (80/20)
#

# Source directory where barcode folders are located
#$unsplit_images_folder = "D:\Visible_Data\2a.Subtracted_SCO_Mean"
$unsplit_images_folder = "D:\Visible_Data\2.Cropped"

# Destination train and val directory
$train_folder = "D:\Visible_Data\3.SplitTrainValTest\Train"
$val_folder = "D:\Visible_Data\3.SplitTrainValTest\Val"
$test_folder = "D:\Visible_Data\3.SplitTrainValTest\Test"

# Recreate child folders to avoid errors
Remove-Item $train_folder -Force -Recurse
Remove-Item $val_folder -Force -Recurse
Remove-Item $test_folder -Force -Recurse
md $train_folder
md $test_folder

# Copy file to randomly selected Train or Val (80/20)
Get-ChildItem $unsplit_images_folder -Recurse | 
    where-object { $_.Extension -in '.jpg','.png' } |
    ForEach-Object { 
    
        $rand_num = Get-Random -Maximum 100

        # 80/20 split
        $dest_folder = If ($rand_num -ge 80) {$test_folder} ElseIf ($rand_num -ge 64) {$val_folder} Else {$train_folder}

        # Preserve last folder [1,2,3,m,ma,4]
        #    -ą is last directory in hierarchy
        $dest_folder += "\" + $_.DirectoryName.split('\')[-1] + "\"
        #echo $_.DirectoryName,$_.DirectoryName.split('\')[-1],$dest_folder
    
        #echo $_.FullName, $dest_folder
        xcopy $_.FullName $dest_folder
    }