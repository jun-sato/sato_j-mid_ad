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
 
 
# Usage
 
"hoge"の基本的な使い方を説明する。

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

 
## eda_json.ipynb
所見文構造化jsonファイルを利用して特定の情報を抽出するファイル。  


## training_DP.py
切り取ってきた臓器画像を用いて異常検知モデルを学習させるファイル。  
model:se-resnext50 (詳細な精度評価は[ここから](https://catkin-resistance-4fa.notion.site/840bbe8525d943b4aa76eba305fc2891))  
pytorch DataParallelを使用。  
```
python training_DP.py --batch_size 16 --datadir /sqfs/work/K22A11/u6b588/jmid/data --num_classes 2 --num_epochs 50 --organ liver --save_model_name weights/liver_seg_baseline_80.pth --segtype seg
```

segtype:'seg'か'square'を使用。squareはbboxで切り取ってくる。

## evaluation.py
学習させたモデルで予測させて精度評価。AUC curveとconfusion_matrixのファイルを出力。
```
python evaluation.py --datadir /mnt/hdd/jmid/data  --organ liver --weight_path ../data/weights/liver_seg_baseline_80_seed4.pth --seed 4 --segtype seg
```

出力されるファイル:
    outputdir上にAUC curveとConfusion matrixのグラフが作成される。
    totalとヘッダーが付いているもの：5fold cvでの結果。
    それ以外：各foldでの結果。


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
    2022/10/30 データの取得や前処理、学習に関するコードを追加。  
    2022/11/30 異常検知に用いるtraining, evaluationのコードを追加。各種コード修正。
    2022/12/13 training, evaluationのコードを修正。

 
# License
ライセンスを明示する。研究室内での使用限定ならその旨を記載。
 
"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
  
"hoge" is Private.
