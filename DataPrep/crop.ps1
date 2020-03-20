# Crop files from SCO-specific directories to a common directory
#   Src: D:\Google Drive\PhD_Data\Raw\SCO*\[1|2|3|4|m|ma]
#   Dest: D:\Visible_Data\2.Cropped\[1|2|3|4|m|ma]


$src_folder = "D:\Google Drive\PhD_Data\Raw\"
$dest_folder = "D:\Visible_Data\2.Cropped\"

$categories = "1","2","3","4","m","ma"

$sco_dir = "SCO1" ,"SCO2","SCO3","SCO4"
$crop_x = 370,300,540,900
$crop_y = 100,130,160,150
$crop_width = 360,360,360,360
$crop_height = 360,360,360,360

# Recreate child folders to avoid errors

foreach ($category in $categories){
    Remove-Item $dest_folder\$category\ -Force -Recurse
    md $dest_folder\$category
}

get-childitem $src_folder'\SCO*' -recurse | 
    where-object { $_.Extension -in '.jpg','.png' } |
    select-object FullName,Name |
    ForEach-Object {
        
        # parse sco number (to select proper crop); 3th element in hierarachy
        $sco = $_.FullName.Split('\')[-3]

        #category 1 of [1|2|3|4|m|ma]
        $category = $_.FullName.Split('\')[-2]

        #echo $sco,$category

        # Proper crop
        $sco_ind = $sco_dir.IndexOf($sco)
        If ($sco_ind -lt 0) {
            echo "SCO not found: ", $sco, $_.FullName
            continue
            }

        $crop_x_this = $crop_x[$sco_ind]
        $crop_y_this = $crop_y[$sco_ind]
        $crop_width_this = $crop_width[$sco_ind]
        $crop_height_this = $crop_height[$sco_ind]

        $dest = $dest_folder + $category + "\" + $_.Name

        #Crop and save
        $cmd = "magick.exe convert -crop " +
                $crop_width_this + "x" + $crop_height_this + "+" + $crop_x_this + "+" + $crop_y_this + 
                " """ + $_.FullName + """ " + $dest
        
        echo $cmd
        iex $cmd

         #magick.exe convert -crop 300x300+283+92 D:/Google Drive/PhD_Data/Raw/SCO1/1/000000005311_4_20191001152033817.jpg D:/Visible_Data/000000005311_4_20191001152033817.jpg
    }