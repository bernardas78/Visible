# Place list of labelled and cropped files to a csv (further used to augment files)
#     Source:  in D:\Visible_Data\2.Cropped\[1|2|3|4|m|ma] contain labelled, cropped images
#     Dest: 'ListLabelledUnsplit_ForAugmentation\[1|2|3|4|m|ma].csv'

$subcategories = "1","2","3","4","m","ma"

foreach ($subcategory in $subcategories)
{
    # empty out results file
    $subcat_filename = 'D:\Google Drive\PhD_Data\Visible_ErrorAnalysis\Relabelling\ListLabelledUnsplit_ForAugmentation\' + $subcategory + '.csv'
    $text | Set-Content $subcat_filename

    $subcat_folder = 'D:\Visible_Data\2.Cropped\' + $subcategory
    get-childitem $subcat_folder |
        where-object { $_.Extension -in '.jpg','.png' } |
        #select-object FullName |
        ForEach-Object {
            # Subcategory, file name
            $_.FullName | Add-Content $subcat_filename
        }
    echo ("Finished subcategory " + $subcategory)
}
