import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

df_lists = sorted(glob.glob('../data/*.csv'))
print(f'all files are {len(df_lists)}')
df = None
for df_path in df_lists:
    if df is None:
        df = pd.read_csv(df_path)
    else:
        tmp = pd.read_csv(df_path)
        df = pd.concat([df,tmp])
df = df.reset_index(drop=True)


start = df_lists[0].split('_')[-1][:-4]
end = df_lists[-1].split('_')[-1][:-4]

df.to_csv(f'../fulllist_{start}_{end}.csv',index=None)


new_df = pd.DataFrame([])

df['ID'] = df['Facility_Code']+df[' Accesion_Number']
ids = df['ID'].unique()
##腹部以外のprotocol を排除
searchfor = ['abd','pancreas','liver','pk','pel','renal','kub','colon','骨盤','腹','体幹']
print(df.shape)
df = df[df[' protocol_name'].fillna('nan').str.lower().str.contains('|'.join(searchfor))]
print(df.shape)

df = df[df[' image_type'].str.lower().str.contains('axial')]
print(df.shape)
###スライス少ないのはスカウト等なので排除
###window_centerがマイナスなのは肺野なので排除。それと一緒に取っている同じスライス数の画像もおそらく胸部の縦隔条件なので排除。
print('start extracting abdomen..')
for id in tqdm(ids):
    tmp = df[df['ID'] == id].reset_index(drop=True)
    num_slices = tmp[' Number_of_Slices']
    window_isminus = tmp[' window_center'].str.contains('-')
    num_slices_low = num_slices<40
    lung_slices = num_slices[window_isminus]
    lung_slices = np.isin(num_slices,lung_slices)

    exclude_slices = num_slices_low | lung_slices
    
    tmp = tmp[~exclude_slices].reset_index(drop=True)

    new_df = pd.concat([new_df,tmp])
print('finish extracting abdomen..')
new_df = new_df.reset_index(drop=True)


print('save files...')
new_df['path'] = new_df['Facility_Code'].astype('str')+'_'+new_df[' Accesion_Number'].astype('int').astype('str')+\
    '_'+new_df[' Series_Number'].fillna(9999).astype('int').astype('str').str.zfill(4)+'_'+new_df[' Number_of_Slices'].astype(int).astype('str').str.zfill(3)+'.nii.gz'

new_df.to_csv(f'../include_abdomen_{start}_{end}.csv',index=False,encoding='utf_8_sig')

new_df['path'].to_csv('../include_abdomen_path_{start}_{end}.csv',index=False,encoding='utf_8_sig')

print('finish correctly')