# Abnormality Detection Based on J-MID Reports

## Features

Train an image classification model by extracting labels from report texts.  
This reduces the labor-intensive annotation work and allows the creation of a large dataset.  
By utilizing J-MID data, we can build a multi-center and the world's largest CT dataset.  
→ Traditional AI medical research (especially in the field of radiology) has struggled with utilizing small datasets, but this approach may solve that problem.

Train an image classification model by segmenting CT images.  
CT images are too large for AI training. Images of 512x512x300 are too large for any GPU. By creating a model that can accurately segment organs, we can reduce the computational load and make it easier to learn the relationship between findings and organs.

## Research Approach

1. Structure all tens of thousands of J-MID finding text data that include the abdomen. The structured data should be linked with information on **which organ's finding** and **what is found**.
2. Collect information on findings for the organs of interest (e.g., liver, gallbladder, pancreas, etc.).
3. Gather CT images with findings of the organs.
4. Use a segmentation model to extract target organs from CT images.
5. Train a classification model using the segmented organs as input and **what is found** information as labels (multiclass & multilabel learning).

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

## Dataset
Refer to [this README](https://github.com/ai-radiol-ou/sato_j-mid_ad/tree/main/download_from_server/) for the process from downloading data and reports from the NII server to segmentation.


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
 
## crop_dataset.py
セグメンテーションされた臓器のうち、特定の臓器を指定する。予測する臓器の周囲だけを抽出してくる。
```bash
python crop_dataset.py --maskdir ../data/pred_1/ --imagedir ../data/renamed_1/ --save_maskdir ../data/liver_pred_1 --save_imagedir ../data/liver_1 --num_threads 20
```

## select_img_from_pred.py
SQUIDでセグメンテーション予測を行い出力されたファイルの元画像をNII取得フォルダから移動させる。  
finished_pred:予測完了したmaskファイル。  
finished_img:予測完了したmaskと対応するimgファイル。  
両者は同数であるはず。  

 

## labeling_{臓器名}.ipynb
所見文構造化jsonファイルを利用して特定の臓器からの情報を抽出するファイル。 



## display_gradcam.ipynb
学習・評価したモデルを使ってモデルの注目部分を可視化する。  
[gradcam](https://github.com/MECLabTUDA/M3d-Cam)と[occlusion_sensitivity](https://docs.monai.io/en/stable/visualize.html#monai.visualize.occlusion_sensitivity.OcclusionSensitivity)を用いたコード。occlusion_sensitivityの方が良い？
occlusion_sisitivityは重要箇所(その部分を隠したときに大きく値が異なる)が青く表示される。
→出力は予測確率であり、重要部分を隠すとそのクラスに属する確率は下がる(negative value)になるから、、、？monaiの[公式のチュートリアル](https://github.com/Project-MONAI/tutorials/blob/main/modules/interpretability/covid_classification.ipynb)をみる限りそんな感じ。

## occlusion_sensitivity.py
occlusion sensitivityのコード。display_gradcam.ipynbの改良版。5-fold CVのアンサンブルの可視化を出力できる。
```bash
python occlusion_sensitivity.py --datadir ../data/ --save_imagedir ../attention_maps/ --organ liver --segtype 25D --seed 0 --backbone tf_efficientnetv2_s_in21ft1k --load_model_name ../data/weights/liver_25D_new_valloss.pth
```


## code/eda.ipynb
NIIサーバー上に置いてるコードの草案。スライス枚数によってグループ分けして、効率よくセグメンテーションが行えるようにする。

# Script for model training/inference
### training_DP.py
切り取ってきた臓器画像を用いて異常検知モデルを学習させるファイル。  
model:se-resnext50 (詳細な精度評価は[ここから](https://catkin-resistance-4fa.notion.site/840bbe8525d943b4aa76eba305fc2891))  
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

 
# Note
 
注意点などがあれば書く
 
# Author
 
作成情報を列挙する
 
* 佐藤淳哉
* 2022/8/21 initial commit  
* 更新情報  
    2022/10/30 データの取得や前処理、学習に関するコードを追加。  
    2022/11/30 異常検知に用いるtraining, evaluationのコードを追加。各種コード修正。  
    2022/12/13 training, evaluationのコードを修正。  
    2022/12/26 visualization(gradcam,occ_sens)のファイルを追加、各臓器のラベリングのフォルダを作成。  
    2023/01/06 serverからのダウンロードファイルを取得(abd_download.sh)。  
    2023/02/08 2.5次元データの学習＆occlusion sensitivityのコードを追加。  
    2023/03/11 2.5次元データの作成コード＆評価コードの追加。  
    2023/06/04 学習に必要なファイルを分離しsrcに収納。trainingファイルも関数ごとに分離した。

 
# License
ライセンスを明示する。研究室内での使用限定ならその旨を記載。
 
This repo is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
  
