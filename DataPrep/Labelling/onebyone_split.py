# Randomly chooses a file and copies to selected category directory
#	Reads from ListFiles.csv (made by ListFiles.ps1)
#	Copies to %GDRIVE%\PhD_Data\Raw\SCO*

import pandas as pd
import numpy as np
from shutil import copy
import os
import math

dest_folder_template = os.environ['GDRIVE'] + "\\PhD_Data\\Raw\\SCO"

print ('Reading file ListFiles.csv...')
df_files = pd.read_csv ('ListFiles.csv', header=None)
print ('Done Reading')

i=0

while (True):
	
	# Uniform random for SCO choice
	sco = math.ceil ( np.random.rand() * 4 )
	
	# filter by sco; get 2nd column (filename)
	df_filenames_sco = df_files.loc [ df_files.iloc[:,0] == sco ] .iloc[:,1]
	
	# Uniform randomly choose a file
	file_index = math.ceil ( np.random.rand() * len(df_filenames_sco) )
	#print ("sco,file_index,len(df_filenames_sco): ", sco,file_index,len(df_filenames_sco))
	sco_filename = df_filenames_sco.iloc [ file_index ]

	# show file
	os.system("explorer \"" + sco_filename + "\"")
	
	# read where to move
	i +=1
	print ("Enter 1,2,3,4,m,ma (total read: ", i, "):")
	category_folder = input()
	if category_folder not in ["1","2","3","4","m","ma"]:
		print ("Unknown category " + category_folder)
	else:

		# copy to dest
		dest_folder = dest_folder_template + str(sco) + '\\' + category_folder + '\\'
		copy (sco_filename, dest_folder)
