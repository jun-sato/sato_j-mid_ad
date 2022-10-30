import pandas as pd
import numpy as np
import os
import shutil
import glob
from tqdm import tqdm

df = pd.read_csv('../server_data/rename.csv')
original_names = df['original_name']
renamed = df['renamed']
pred_dir = '../data/finish_pred/'
save_dir = '../data/finish_img/'
default_dir = '../download_from_server/data/abd_20220101-20220605/'

pred_files = glob.glob(os.path.join(pred_dir,'*.nii.gz'))
img_predicted_files = [os.path.basename(p).split('.nii')[0]+'_0000.nii.gz' for p in pred_files]
count=0
for orig, rename in tqdm(zip(original_names,renamed)):
    if rename in img_predicted_files:
        shutil.move(os.path.join(default_dir,orig),os.path.join(save_dir,rename))
        count+=1
print(f'finished! The number of moved file is {count}')