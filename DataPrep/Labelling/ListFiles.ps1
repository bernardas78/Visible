# Place list of files to a csv (further used to label files)
#     Source: SCO1, ..., SCO4 in J:\AK Dropbox\Data\Upload\pictures\ contain raw images
#     Dest: 'ListFiles.csv'

# empty out results file
$text | Set-Content 'ListFiles.csv'


get-childitem 'J:\AK Dropbox\Data\Upload\pictures\SCO*' -recurse | 
    where-object { $_.Extension -in '.jpg','.png' } |
    select-object FullName |
    ForEach-Object {

        # parse sco number (later used to crop); 5th element in hierarachy; 3rd cha
        $sco = $_.FullName.Split('\')[5][3]

        # SCO number, file name
        $sco + ',' + $_.FullName | Add-Content 'ListFiles.csv'
    }
