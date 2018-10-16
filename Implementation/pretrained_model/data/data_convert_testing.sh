#!/bin/sh

red=`tput setaf 1`
green=`tput setaf 2`
blue=`tput setaf 4`
reset=`tput sgr0`

if [ $# -ne 2 ] 
   then echo "${red}Enter the path of the directory with the test dataset${reset}"
   		read path_dataset 
   		echo "${red}Enter the path of the directory for the final datasets (tensorflow)${reset}"
   		read path_tensor
else
    path_dataset=$1 
    echo ${path_dataset};
    path_tensor=$2 
    echo ${path_tensor};
fi

cat ${path_dataset}/../testing.txt | cut -d ' ' -f2,13,14,15,16 > temp.txt

# --gender:     1 for male, 2 for female
# --smile:      1 for smiling, 2 for not smiling
# --glasses:    1 for wearing glasses, 2 for not wearing glasses.
# --head pose:  1 for left profile, 2 for left, 3 for frontal, 4 for right, 5 for right profile
mkdir ${path_tensor}/dataset_gender
mkdir ${path_tensor}/dataset_smile
mkdir ${path_tensor}/dataset_glasses
mkdir ${path_tensor}/dataset_head_pose
mkdir ${path_tensor}/dataset_all


mkdir ${path_tensor}/dataset_gender/male
mkdir ${path_tensor}/dataset_gender/female

mkdir ${path_tensor}/dataset_smile/smiling
mkdir ${path_tensor}/dataset_smile/not_smiling

mkdir ${path_tensor}/dataset_glasses/wearing_glasses
mkdir ${path_tensor}/dataset_glasses/not_wearing_glasses

mkdir ${path_tensor}/dataset_head_pose/left_profile
mkdir ${path_tensor}/dataset_head_pose/left
mkdir ${path_tensor}/dataset_head_pose/frontal
mkdir ${path_tensor}/dataset_head_pose/right
mkdir ${path_tensor}/dataset_head_pose/right_profile

mkdir ${path_tensor}/dataset_all/male
mkdir ${path_tensor}/dataset_all/female
mkdir ${path_tensor}/dataset_all/smiling
mkdir ${path_tensor}/dataset_all/not_smiling
mkdir ${path_tensor}/dataset_all/wearing_glasses
mkdir ${path_tensor}/dataset_all/not_wearing_glasses
mkdir ${path_tensor}/dataset_all/left_profile
mkdir ${path_tensor}/dataset_all/left
mkdir ${path_tensor}/dataset_all/frontal
mkdir ${path_tensor}/dataset_all/right
mkdir ${path_tensor}/dataset_all/right_profile

rm gender.sh
rm smile.sh
rm glasses.sh
rm head_pose.sh
rm all.sh


while read name_file gender smile wearing_glasses head_pose; do
  
  echo "${name_file} ${gender} ${smile} ${wearing_glasses} ${head_pose}"
  name=`basename ${path_dataset}/${name_file}`
  echo "${name}"

  if [[ $gender == 1* ]]; then
    echo "male"
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_gender/male/${name}">>gender.sh
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_all/male/${name}">>all.sh
  fi
  if [[ $gender == 2* ]]; then
    echo "female"
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_gender/female/${name}">>gender.sh
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_all/female/${name}">>all.sh
  fi


  if [[ $smile == 1* ]]; then
    echo "smiling"
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_smile/smiling/${name}">>smile.sh
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_all/smiling/${name}">>all.sh
  fi
  if [[ $smile == 2* ]]; then
    echo "not smiling"
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_smile/not_smiling/${name}">>smile.sh
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_all/not_smiling/${name}">>all.sh
  fi


  if [[ $wearing_glasses == 1* ]]; then
    echo "glasses"
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_glasses/wearing_glasses/${name}">>glasses.sh
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_all/wearing_glasses/${name}">>all.sh
  fi
  if [[ $wearing_glasses == 2* ]]; then
    echo "no glasses"
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_glasses/not_wearing_glasses/${name}">>glasses.sh
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_all/not_wearing_glasses/${name}">>all.sh
  fi



  if [[ $head_pose == 1* ]]; then
    echo "left profile"
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_head_pose/left_profile/${name}">>head_pose.sh
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_all/left_profile/${name}">>all.sh
  fi
  if [[ $head_pose == 2* ]]; then
    echo "left"
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_head_pose/left/${name}">>head_pose.sh
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_all/left/${name}">>all.sh
  fi
  if [[ $head_pose == 3* ]]; then
    echo "frontal"
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_head_pose/frontal/${name}">>head_pose.sh
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_all/frontal/${name}">>all.sh
  fi
  if [[ $head_pose == 4* ]]; then
    echo "right"
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_head_pose/right/${name}">>head_pose.sh
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_all/right/${name}">>all.sh
  fi
  if [[ $head_pose == 5* ]]; then
    echo "right profile"
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_head_pose/right_profile/${name}">>head_pose.sh
    echo "cp ${path_dataset}/${name} ${path_tensor}/dataset_all/right_profile/${name}">>all.sh
  fi

done < temp.txt

rm temp.txt

bash gender.sh
bash smile.sh
bash wearing_glasses.sh
bash head_pose.sh
bash all.sh