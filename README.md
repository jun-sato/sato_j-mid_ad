# Annotation-free multi-organ anomaly detection in abdominal CT using free-text radiology reports

## Features

Train an image classification model by extracting labels from radiology free-text reports.  
This reduces the labor-intensive annotation work and allows the creation of a large dataset.  
A multi-organ segmentation model and an information extraction schema were used to extract specific organ images and disease information, CT images and radiology reports, respectively, which were used to train a multiple-instance learning model for anomaly detection.

## Requirements

Required Libraries:

* monai
* SimpleITK
* Pytorch
* Timm
* matplotlib

## Installation

Here is how to install the listed libraries.  
For training nnUNet, refer to the [official GitHub](https://github.com/MIC-DKFZ/nnUNet).  
Install PyTorch suitable for your GPU environment from the [official site](https://pytorch.org/get-started/locally/). This example uses pytorch==1.13.1:  
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
```bash
pip install monai
pip install SimpleITK
pip install timm
```

If you are using conda, [recreating the environment with conda](https://qiita.com/nshinya/items/cb1cffabc3305c907bc5) is convenient.

## model weights
The pretrained weights used in this study (Organ segmentation, anomaly detection, dataset size experiment) are available from [google drive](https://drive.google.com/drive/folders/17DgUVCo1We4EHM6PSeJ4ChtNZU5zhAJV?usp=sharing)



```
data  
 |-----liver
 |        |---jmid_0000000_0000.nii.gz  
 |        |---jmid_0000001_0000.nii.gz  
 |  
 |-----pancreas  
 |        |---jmid_0000000_0000.nii.gz  
 |        |---jmid_0000001_0000.nii.gz    
 |-----spleen
 |        |---jmid_0000000_0000.nii.gz  
 |        |---jmid_0000001_0000.nii.gz  
 |
```
### Abnormality Labels
```
Liver: [‘cyst’, ‘fatty_liver’, ‘bile_duct_dilation’, ‘SOL’, ‘deformation’, ‘calcification’, ‘pneumobilia’, ‘other_abnormality’, ‘nofinding’]
Gallbladder: [‘SOL’, ‘enlargement’, ‘deformation’, ‘gallstone’, ‘wall_thickening’, ‘polyp’, ‘other_abnormality’, ‘nofinding’]
Pancreas: [‘cyst’, ‘SOL’, ‘enlargement’, ‘atrophy’, ‘calcification’, ‘pancreatic_duct_dilation/atrophy’, ‘other_abnormality’, ‘nofinding’]
Spleen: [‘cyst’, ‘SOL’, ‘deformation’, ‘calcification’, ‘other_abnormality’, ‘nofinding’]
Kidney: [‘cyst’, ‘SOL(including_complicated_cyst)’, ‘enlargement’, ‘atrophy’, ‘deformation’, ‘calcification’, ‘other_abnormality’, ‘nofinding’]
Adrenal_gland: [‘SOL’, ‘enlargement’, ‘fat’, ‘calcification’, ‘other_abnormality’, ‘nofinding’]
Esophagus: [‘mass’, ‘hernia’, ‘dilation’, ‘other_abnormality’, ‘nofinding’]
```


 
# Usage

**丁寧かつ簡潔に、初めて見た人でも理解できる様に**

* データセットの説明。どの位置に置き、どのような形式にしておけば良いか？
* 主要なファイルの説明。それぞれどのような関数があり、何ができるか？
* コード実行の手順を記載。どうしたら目的の成果(モデルの学習や成果物の保存など)が得られるか。
 


# Script for model training/inference
### training_DP.py
切り取ってきた臓器画像を用いて異常検知モデルを学習させるファイル。  
pytorch DataParallelを使用。  
```
python training_DP.py --batch_size 16 --datadir /sqfs/work/K22A11/u6b588/jmid/data --num_classes 2 --num_epochs 50 --organ liver --save_model_name weights/liver_seg_baseline_80.pth --segtype seg
```

segtype:'seg'か'square'を使用。squareはbboxで切り取ってくる。

## training_25D.py
2.5次元CNNを使ったモデルで異常検知モデルを学習させるファイル。

## evaluation.py(evaluation_25D.py)
学習させたモデルで予測させて精度評価。AUC curveとconfusion_matrixのファイルを出力。
```
python evaluation.py --datadir /mnt/hdd/jmid/data  --organ liver --weight_path ../data/weights/liver_seg_baseline_80_seed4.pth --seed 4 --segtype seg
```

```
python evaluation_25D.py --datadir /mnt/hdd/jmid/data  --organ liver --weight_path ../data/weights/liver_25D_new_valloss.pth --segtype 25D --outputdir ../result_eval/
```

出力されるファイル:
    outputdir上にAUC curveとConfusion matrixのグラフが作成される。
    totalとヘッダーが付いているもの：5fold cvでの結果。
    それ以外：各foldでの結果。

## dataset.py
データセットの定義関数

## models.py
CNNモデルの定義関数

## transforms.py
augmentation/transformの定義関数

## loss.py
loss関数の定義関数

## utils.py
コードに使ういろいろな関数。

 
 
# Author information
 
作成情報を列挙する
 
* Junya Sato
* 2022/8/21 initial commit  

 
# License
 
This code is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
  
