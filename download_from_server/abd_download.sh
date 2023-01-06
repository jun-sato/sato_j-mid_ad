#!/bin/bash
 
for file in `awk 'NR>=2{print $1}' include_abdomen_path.csv`
do
  scp -i ~/.ssh/nii usropsaiE1@10.1.10.53:/home/users/Share3/radiology/nifti_and_findings_daily/CT/nifti/2022/*/$file /mnt/hdd/jmid/download_from_server/data/abd_20220101-20220605/ 
done
