import os
import shutil
import glob 
import pandas as pd
from tqdm import tqdm
data_dir = '../data1/pred_*_*/'
finished_list = []
finished_paths = glob.glob(os.path.join(data_dir,'*.nii.gz'))
for path in tqdm(finished_paths):
    base = os.path.basename(path).split('.nii')
    #num = int(base[0].split('_')[-1])//2000 + 1
    #img_dir = '../data/renamed_'+str(num)
    img_dir  = os.path.dirname(path).replace('pred','remain')
    img_path = base[0] + '_0000.nii.gz'
    finished_list.append(img_path)
    shutil.move(os.path.join(img_dir,img_path),os.path.join('../data/finish_img2/',img_path))
pd.DataFrame(finished_list).to_csv('finished4.csv',index=None)
