# Merges Subsets Category1    |    Category2
#                   1,  | 2,3,m,ma | 4

# Ucomment one of the lines bellow
$invisible_categories = '1'
#$invisible_categories = '1','2'
#$invisible_categories = '1','2',3
#$invisible_categories = '1','ma'
#$invisible_categories = '1','2','ma'
#$invisible_categories = '1','2',3,'ma'
#$invisible_categories = '1','ma','m'
#$invisible_categories = '1','2','ma','m'
#$invisible_categories = '1','2',3,'ma','m'

# Source directory where barcode folders are located
$unsplit_images_folder = "D:\Visible_Data\2.Cropped"

# Destination train and val directory
$train_folder = "c:\TrainAndVal_Visible\Train"
$val_folder = "c:\TrainAndVal_Visible\Val"

# Recreate child folders to avoid errors
Remove-Item $train_folder\Invisible -Force -Recurse
Remove-Item $train_folder\Visible -Force -Recurse
Remove-Item $val_folder\Invisible -Force -Recurse
Remove-Item $val_folder\Visible -Force -Recurse
md $train_folder\Invisible
md $train_folder\Visible
md $val_folder\Invisible
md $val_folder\Visible

# Copy file to randomly selected Train or Val (80/20)
Get-ChildItem $unsplit_images_folder -Recurse | 
    where-object { $_.Extension -in '.jpg','.png' } |
    ForEach-Object { 
    
        $rand_num = Get-Random -Maximum 100

        # 80/20 split
        $dest_folder = If ($rand_num -ge 80) {$val_folder} Else {$train_folder}

        # Merge Subsets 1,  | 2,3,m,ma | 4
        #    -2 is last directory in hierarchy
        $dest_folder += If ( $_.DirectoryName.split('\')[-1] -in $invisible_categories) {'\Invisible'} Else {'\Visible'}
        #echo $_.DirectoryName,$_.DirectoryName.split('\')[-1],$dest_folder
    
        #echo $_.FullName, $dest_folder
        copy $_.FullName $dest_folder
    }