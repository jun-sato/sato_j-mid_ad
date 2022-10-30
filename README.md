# J-MID report-based 異常検知
 

* 研究の内容
* 既に論文などで公開されているならその情報
* 参考にしたリポジトリなどあればその情報

 
# Features
 
**ラベルを所見文から抽出して**画像分類モデルを学習する。  
これにより、アノテーションの労働力的負担が減少し、大きなデータセットを作成できる。  
更に、J-MIDデータを利用することにより、多施設かつ世界最大規模のCTデータセットを構築できる。  
→今までのAI医療研究(特に放射線領域)は少ないデータを如何に利用するかが課題だったが、それらを解決できるかもしれない。  

**CT画像をセグメンテーションして**画像分類モデルを学習させる。  
CT画像はAI学習には大きすぎる。512☓512☓300の画像はどのGPUにも載らない。臓器を正確にセグメンテーションできるモデルを作成することにより、計算量を落とし、所見と臓器との関連を学習させやすくする。  

## 研究方針
1. 腹部を含んだ数十万件のJ-MID所見分データをすべて構造化する。構造化したデータには、**どの臓器の所見か**という情報と、**何があるか**という情報が紐付いている。
2. 注目する臓器(まず肝胆膵？)の所見の情報を集める。
3. 臓器の所見のあるCT画像を収集してくる。
4. CT画像からセグメンテーションモデルを使い対象臓器を取ってくる。
5. セグメンテーションした臓器をinput,**何があるか**という情報をlabelとして分類モデルを学習する。(multiclass & multilabel learning)

 
# Requirement
 
"hoge"を動かすのに必要なライブラリなどを列挙する
 
* monai
* simpleitk
 
# Installation
 
Requirementで列挙したライブラリなどのインストール方法を説明する
 nnUNetの学習方法は[公式github](https://github.com/MIC-DKFZ/nnUNet)を参照する。

```bash
pip install monai
pip install SimpleITK
```

condaを使っている場合は[condaによる環境再現](https://qiita.com/nshinya/items/cb1cffabc3305c907bc5)が便利です。
 
# Dataset

```
dataset  
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

 
 
# Usage
 
"hoge"の基本的な使い方を説明する。

**丁寧かつ簡潔に、初めて見た人でも理解できる様に**

* データセットの説明。どの位置に置き、どのような形式にしておけば良いか？
* 主要なファイルの説明。それぞれどのような関数があり、何ができるか？
* コード実行の手順を記載。どうしたら目的の成果(モデルの学習や成果物の保存など)が得られるか。
 
## crop_dataset.py
```bash
##セグメンテーションされた臓器のうち、特定の臓器を指定する。予測する臓器の周囲だけを抽出してくる。
python crop_dataset.py --maskdir ../data/pred_1/ --imagedir ../data/renamed_1/ --save_maskdir ../data/liver_pred_1 --save_imagedir ../data/liver_1 --num_threads 20
```

## select_img_from_pred.py
SQUIDでセグメンテーション予測を行い出力されたファイルの元画像をNII取得フォルダから移動させる。  
finished_pred:予測完了したmaskファイル。  
finished_img:予測完了したmaskと対応するimgファイル。  
両者は同数であるはず。  

 
## eda_json.ipynb
所見文構造化jsonファイルを利用して特定の情報を抽出するファイル。  


## training_DP.py
切り取ってきた臓器画像を用いて異常検知モデルを学習させるファイル。  
model:resnet18  
DataParallel  
```
python training_DP.py --organ pancreas --num_epochs 5 --batch_size 2 --datadir /mnt/hdd/jmid/data/
```

## code/extract_abdomen.py
NIIサーバー上から患者情報のcsvファイルをダウンロード。腹部に対応するCT所見をcsv情報から絞り込んでくる。  
include_abdomen_{開始期間}_{終了期間}.csvのファイルが出力される。また、pathの情報のファイルも出力される。


## code/eda.ipynb
NIIサーバー上に置いてるコードの草案。スライス枚数によってグループ分けして、効率よくセグメンテーションが行えるようにする。




 
# Note
 
注意点などがあれば書く
 
# Author
 
作成情報を列挙する
 
* 佐藤淳哉
* 2022/8/21 initial commit  
* 更新情報
    2022/10/30 実効コードを追加。
 
# License
ライセンスを明示する。研究室内での使用限定ならその旨を記載。
 
"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
  
"hoge" is Private.
