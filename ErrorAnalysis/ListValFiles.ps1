# Place list of validation files to a csv (further used to do error analysis)
#     Source:  in C:\TrainAndVal_6classes\[Train|Val]\[1|2|3|4|m|ma] validation images that a model was trained on
#     Dest: 'ListValFiles.csv'

# empty out results file
$text | Set-Content 'ListValFiles.csv'


get-childitem 'C:\TrainAndVal\Test\*' -recurse |
    where-object { $_.Extension -in '.jpg','.png' } |
    #select-object FullName |
    ForEach-Object {

        # parse test/val and sub-category
        $subcategory = $_.DirectoryName.Split('\')[-1]
        #echo $_.DirectoryName
        #echo $train_or_test
        #echo $subcategory

        # sub-category, file name
        $subcategory + ',' + $_.FullName | Add-Content 'ListValFiles.csv'
    }
