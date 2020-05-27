# Deletes a percentage of data from "3.SplitTrainValTest\[Train|Val]\[1|2|3|4|m|ma]
#     Don't delete Test set to evaluate to ground truth as precisely as possible

# delete probability (6K is original dataset size)
$delete_prob = 1-4500/6000 

$split_images_folder = "D:\Visible_Data\3.SplitTrainValTest"

$sets = "Train","Val"
$categories = "1","2","3","4","m","ma"

#$num=0
foreach ($set in $sets){
    foreach ($category in $categories){

        # Delete file based on random probability
        $subcat_folder = $split_images_folder+"\"+$set+"\"+$category
        Get-ChildItem $subcat_folder | 
            where-object { $_.Extension -in '.jpg','.png' } |
            ForEach-Object { 
    
                $rand_num = Get-Random -Maximum 10000

                if ($rand_num -lt $delete_prob*10000){
                    del $_.FullName
                    #echo $_.FullName
                    #echo "Deleting "+$num
                    #$num+=1
                    }
            }
    }
}

