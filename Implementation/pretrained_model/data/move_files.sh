#!/bin/sh

red=`tput setaf 1`
green=`tput setaf 2`
blue=`tput setaf 4`
reset=`tput sgr0`


if [ $# -ne 2 ] 
   then echo "${red}Enter the path of the directory with the TEST dataset folder ${reset}"
   		read path_dataset 
   		echo "${red}Enter the path of the directory with the TRAINING dataset folder${reset}"
   		read path_training

else
    path_dataset=$1 
    echo ${path_dataset};
    path_training=$2 
    echo ${path_training};
fi

find ${path_dataset}/smiling -type f > temp_smile.txt
while read name_file ; do
  base_name=`basename ${name_file}`
  cp ${path_dataset}/smiling/${base_name} ${path_training}/smiling/test_${base_name}

done < temp_smile.txt

find ${path_dataset}/not_smiling -type f > temp_not_smile.txt
while read name_file ; do
  base_name=`basename ${name_file}`
  cp ${path_dataset}/not_smiling/${base_name} ${path_training}/not_smiling/test_${base_name}

done < temp_not_smile.txt

rm temp_smile.txt
rm temp_not_smile.txt


find ${path_dataset}/male -type f > temp_male.txt
while read name_file ; do
  base_name=`basename ${name_file}`
  cp ${path_dataset}/male/${base_name} ${path_training}/male/test_${base_name}

done < temp_male.txt


find ${path_dataset}/female -type f > temp_female.txt
while read name_file ; do
  base_name=`basename ${name_file}`
  cp ${path_dataset}/female/${base_name} ${path_training}/female/test_${base_name}

done < temp_female.txt

rm temp_male.txt
rm temp_female.txt