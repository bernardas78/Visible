# Place list of validation files to a csv (further used to do error analysis)
#     Source:  in C:\TrainAndVal_6classes\[Train|Val]\[1|2|3|4|m|ma] validation images that a model was trained on
#     Dest: 'ListValFiles.csv'

# empty out results file
$text | Set-Content $env:GDRIVE'\PhD_Data\Visible_ErrorAnalysis\Relabelling\ListFiles_Relabelled.bat'

$relabelled_folder = '%GDRIVE%\PhD_Data\Raw_Relabelled_1\'


# get-childitem 'C:\TrainAndVal_6classes\Val\*' -recurse |
get-childitem 'D:\Visible_Data\RelabelReady\*' -recurse |
    where-object { $_.Extension -in '.jpg','.png' } |
    #select-object FullName |
    ForEach-Object {

        # parse test/val and sub-category
        #$subcategory = $_.DirectoryName.Split('\')[-1]
        #echo $_.DirectoryName
        #echo $train_or_test
        #echo $subcategory

        # Parse from structure \<new_class>\<old_class>_<xx>_<xx>_<sco>_<orig_filename>
        $newclass = $_.FullName.Split('\')[-2]
        $oldclass = $_.Name.Split('_')[0]
        $sco = $_.Name.Split('_')[3]
        # orig filename contains underscores, so concatenate back
        $filename_arr = $_.Name.Split('_')
        $origfilename = $filename_arr[4..($filename_arr.length-1)] -join "_"

        # sub-category, file name
        $command = "move """ + $relabelled_folder + "SCO" + $sco + "\" + $oldclass + "\" + $origfilename + """ """ + $relabelled_folder + "SCO" + $sco + "\" + $newclass + """"
        echo $command | Add-Content $env:GDRIVE'\PhD_Data\Visible_ErrorAnalysis\Relabelling\ListFiles_Relabelled.bat'
    }
