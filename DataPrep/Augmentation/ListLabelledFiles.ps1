# Place list of labelled and cropped files to a csv (further used to augment files)
#     Source:  in D:\Visible_Data\3.SplitTrainVal\[Train|Val]\[1|2|3|4|m|ma] contain labelled, cropped images
#     Dest: 'ListLabelledFiles.csv'

# empty out results file
$text | Set-Content 'ListLabelledFiles.csv'


# Experiments with dataset size
#get-childitem 'D:\Visible_Data\3.SplitTrainValTest_3647\*' -recurse |

get-childitem 'D:\Visible_Data\3.SplitTrainValTest\*' -recurse |
    where-object { $_.Extension -in '.jpg','.png' } |
    #select-object FullName |
    ForEach-Object {

        # parse test/val and sub-category
        $train_or_test = $_.DirectoryName.Split('\')[-2]
        $subcategory = $_.DirectoryName.Split('\')[-1]
        #echo $_.DirectoryName
        #echo $train_or_test
        #echo $subcategory

        # SCO number, file name
        $train_or_test + "," + $subcategory + ',' + $_.FullName | Add-Content 'ListLabelledFiles.csv'
    }
