# Crop files from SCO-specific directories to a common directory
#   Src: %GDRIVE%\PhD_Data\Raw\SCO*\[1|2|3|4|m|ma]
#   Dest: D:\Visible_Data\2.Cropped\[1|2|3|4|m|ma]


$src_folder = $env:GDRIVE+"\PhD_Data\Raw\"
$dest_folder = "D:\Visible_Data\2.Cropped_BySCO\"

$categories = "1","2","3","4","m","ma"

$sco_dir = "SCO1" ,"SCO2","SCO3","SCO4"

#$crop_x = 370,300,540,900
#$crop_y = 100,130,160,150
#$crop_width = 360,360,360,360
#$crop_height = 360,360,360,360

class CropProperties {
    [string]$sco
    [string]$cutoffdatetime
    [int]$x
    [int]$y
    [int]$width
    [int]$height;
}

# Camera was moved at certain times; take proper crops
$crops = @()
$crops += [CropProperties]@{ sco="SCO1";cutoffdatetime=20190909090000;x=235;y=100;width=360;height=360 } 
$crops += [CropProperties]@{ sco="SCO1";cutoffdatetime=20190929110000;x=370;y=100;width=360;height=360 } 
$crops += [CropProperties]@{ sco="SCO1";cutoffdatetime=20191009000000;x=290;y=100;width=360;height=360 }
$crops += [CropProperties]@{ sco="SCO1";cutoffdatetime=20191208150000;x=520;y=100;width=360;height=360 }
$crops += [CropProperties]@{ sco="SCO1";cutoffdatetime=20191210110000;x=290;y=100;width=360;height=360 }
$crops += [CropProperties]@{ sco="SCO1";cutoffdatetime=20200101000000;x=175;y=100;width=360;height=360 }

$crops += [CropProperties]@{ sco="SCO2";cutoffdatetime=20190907134500;x=300;y=130;width=360;height=360 }
$crops += [CropProperties]@{ sco="SCO2";cutoffdatetime=20200101000000;x=255;y=130;width=360;height=360 }

$crops += [CropProperties]@{ sco="SCO3";cutoffdatetime=20200101000000;x=540;y=160;width=360;height=360 }

$crops += [CropProperties]@{ sco="SCO4";cutoffdatetime=20200101000000;x=900;y=150;width=360;height=360 }


# Recreate child folders to avoid errors

foreach ($sco in $sco_dir){
    Remove-Item $dest_folder\$sco\ -Force -Recurse
    md $dest_folder\$sco
    foreach ($category in $categories){
        #Remove-Item $dest_folder\$category\ -Force -Recurse
        md $dest_folder\$sco\$category
    }
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

        $filedatetime = $_.Name.Split('_')[-1].Substring(0,14)  # files named like 477016319917_9_20191210155832438.jpg

        $found=$false
        $crops.getEnumerator() | Where-Object -FilterScript {$_.sco -eq $sco} | Sort-Object -Property cutoffdatetime | foreach {
            if ($_.cutoffdatetime -gt $filedatetime -and -not $found){
                $found=$true
                $crop_x_this = $_.x
                $crop_y_this = $_.y
                $crop_width_this = $_.width
                $crop_height_this = $_.height
                }
                #Write-Host($_.cutoffdatetime)
            }

        #$sco_ind = $sco_dir.IndexOf($sco)
        #If ($sco_ind -lt 0) {
        #    echo ("SCO not found: " + $sco + " file: " + $_.FullName)
        #    return # instead "continue", "return" only breaks from iteration: https://stackoverflow.com/questions/7760013/why-does-continue-behave-like-break-in-a-foreach-object
        #    }

        #$crop_x_this = $crop_x[$sco_ind]
        #$crop_y_this = $crop_y[$sco_ind]
        #$crop_width_this = $crop_width[$sco_ind]
        #$crop_height_this = $crop_height[$sco_ind]

        # Save by SCO
        $dest = $dest_folder + $sco + "\" + $category + "\" + $_.Name

        #Temporary: split by sco; all classes to 1 dir; start with datetime
        #$dest = $dest_folder + $sco + "\" + $_.Name.Split('_')[-1].Substring(0,14) + "." + $_.Name.Split('.')[-1]

        #Crop and save
        $cmd = "magick.exe convert -crop " +
                $crop_width_this + "x" + $crop_height_this + "+" + $crop_x_this + "+" + $crop_y_this + 
                " """ + $_.FullName + """ " + $dest

        echo $cmd
        iex $cmd

         #magick.exe convert -crop 300x300+283+92 D:/Google Drive/PhD_Data/Raw/SCO1/1/000000005311_4_20191001152033817.jpg D:/Visible_Data/000000005311_4_20191001152033817.jpg
    }