###スライス数ごとに画像を分けて、別々のフォルダに保存する
###フォルダ内のファイル数を一定に保つ


import pandas as pd
import glob
import os
import shutil
import argparse
from tqdm import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='In order to remove unnecessary background, crop images based on segmenation labels.')
    parser.add_argument('--datadir', default="../data/abd_20220101-20220605/",
                        help='dafault data directory')
    parser.add_argument('--start_id', default=0, type=int,
                        help='a number to start rename , (default: 0)')
    parser.add_argument('--num_threads', default=20, type=int,
                        help='number of threds to process , (default: 20)')

    args = parser.parse_args()
    print(args)
    datadir = args.datadir
    start_id = args.start_id

    img_paths = sorted(glob.glob(os.path.join(datadir,'*.nii.gz')))
    print(len(img_paths))
    num_imgs = len(img_paths)
    num_slices = [int(i.split('_')[-1].split('.')[0]) for i in img_paths]

    new_paths = [f'jmid_{str(i+start_id).zfill(7)}_0000.nii.gz' for i in range(num_imgs)]
    img_paths_base = [os.path.basename(p) for p in img_paths]

    df = pd.DataFrame([img_paths_base,new_paths]).T
    df.columns = ['original_name','renamed']
    df.to_csv('rename_20211001-20211231.csv',index=False)

    dir_id = 0

    for i in range(9):
        os.makedirs(f'../data1/remain_{str(i)}_0',exist_ok=True)

    for num_slice,img_path,new_path in tqdm(zip(num_slices,img_paths,new_paths)):
        if num_slice>=450:
            print(f'large files found. skip img is {img_path}')
            continue
        else:
            out_dir = '../data1/remain_' + str(8-num_slice//50) + '_0'
            shutil.copyfile(img_path,os.path.join(out_dir,new_path))
    print('data copied!')
    ####
    ####8,7→5000 6→2000, 5→1000 4→700 3→400 2→300 1,0→100
    for i in range(9):
        if i == 8 or i == 7:
            upper = 5000
        elif i == 6:
            upper = 2000
        elif i == 5:
            upper = 1000
        elif i == 4:
            upper = 700
        elif i == 3:
            upper = 400
        elif i == 2:
            upper = 300
        else:
            upper = 100


        files = glob.glob(f'../data1/remain_{str(i)}_0/*nii.gz')
        num_files = len(files)
        num_folder = num_files//upper
        for n in range(num_folder):
            os.makedirs(f'../data1/remain_{str(i)}_{str(n+1)}',exist_ok=True)
        for k,path in enumerate(files):
            dir_id = k // upper
            if dir_id >= 1:
                shutil.move(path,os.path.join(f'../data1/remain_{str(i)}_{str(dir_id)}',os.path.basename(path)))
    print('file moved.')
