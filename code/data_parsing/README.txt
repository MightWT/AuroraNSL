# auroraproject
find the files in a desired fold 
for each file with extension (.NC, must all capital), work on the following process:
open the file with .NC extension
open an variable named "NORTH_GEOGRAPHIC_LATITUDE" in this file
open another variable named "NORTH_GEOGRAPHIC_LONGITUDE" in this file
open another variable named "NORTH_POLAR_GEOGRAPHIC_LATITUDE" in this file
open another variable named "NORTH_POLAR_GEOGRAPHIC_LONGITUDE" in this file
open another variable named "SOUTH_GEOGRAPHIC_LATITUDE" in this file
open another variable named "SOUTH_GEOGRAPHIC_LONGITUDE" in this file
open another variable named "SOUTH_POLAR_GEOGRAPHIC_LATITUDE" in this file
open another variable named "SOUTH_POLAR_GEOGRAPHIC_LONGITUDE" in this file
Read an attribute named "STOPPING_TIME"
write the data from NORTH_GEOGRAPHIC_LATITUDE into a csv file named "Auroral boundary LATITUDE and LONGITUDE" + the value in attribute "STOPPING_TIME". make sure put this data in column B and one data per row
write the data from NORTH_GEOGRAPHIC_LONGITUDE into the same csv file. make sure put this data in column C and one data per row
write the data from NORTH_POLAR_GEOGRAPHIC_LATITUDE into the same  csv file. make sure put this data in column d and one data per row
write the data from NORTH_POLAR_GEOGRAPHIC_LONGITUDE into the same  csv file. make sure put this data in column e and one data per row
write the data from SOUTH_GEOGRAPHIC_LATITUDE into a csv file named this file's name without extension. make sure put this data in column f and one data per row
write the data from SOUTH_GEOGRAPHIC_LONGITUDE into the same csv file. make sure put this data in column g and one data per row
write the data from SOUTH_POLAR_GEOGRAPHIC_LATITUDE into the same  csv file. make sure put this data in column h and one data per row
write the data from SOUTH_POLAR_GEOGRAPHIC_LONGITUDE into the same  csv file. make sure put this data in column i and one data per row
generate an index for each data from NORTH_GEOGRAPHIC_LATITUDE starting from 1 and increase by 1 for each data and put those data in column A in the same csv file.
make sure each column has a different number of data, so handling columns with different numbers of data by adding commas for the missing values. Define a function to simplify this process 
write all csv file into a folder named "Auroral boundary LATITUDE and LONGITUDE"
