# Merges Subsets Category1    |    Category2
#                   1,  | 2,3,m,ma | 4

# Ucomment one of the lines bellow
#$invisible_categories = '1'
#$invisible_categories = '1','2'
#$invisible_categories = '1','2',3
$invisible_categories = '1','ma'
#$invisible_categories = '1','2','ma'
#$invisible_categories = '1','2',3,'ma'
#$invisible_categories = '1','ma','m'
#$invisible_categories = '1','2','ma','m'
#$invisible_categories = '1','2',3,'ma','m'

# Source directory
$src_folder = "D:\Visible_Data\3.SplitTrainValTest"

# Destination 2 class folder
$dest_folder = "D:\Visible_Data\3.SplitTrainValTest_2class"

# Recreate child folders to avoid errors
Remove-Item $dest_folder\Train -Force -Recurse
Remove-Item $dest_folder\Val -Force -Recurse
Remove-Item $dest_folder\Test -Force -Recurse
md $dest_folder\Train
md $dest_folder\Train\Invisible
md $dest_folder\Train\Visible
md $dest_folder\Val
md $dest_folder\Val\Invisible
md $dest_folder\Val\Visible
md $dest_folder\Test
md $dest_folder\Test\Invisible
md $dest_folder\Test\Visible

# Copy files to Invisible or Visible
Get-ChildItem $src_folder -Recurse |
    where-object { $_.Extension -in '.jpg','.png' } |
    ForEach-Object { 
    
        # Merge Subsets 1,  | 2,3,m,ma | 4
        #    -2 = Train|Val|Test
        $dest_sub_folder = $dest_folder + "\" + $_.DirectoryName.split('\')[-2] + "\"
        #    -1 = 1|2|3|4|m|ma
        $dest_sub_folder += If ( $_.DirectoryName.split('\')[-1] -in $invisible_categories) {'\Invisible'} Else {'\Visible'}
        #echo $_.DirectoryName,$_.DirectoryName.split('\')[-1],$dest_folder
    
        #echo $_.FullName, $dest_folder
        copy $_.FullName $dest_sub_folder
    }