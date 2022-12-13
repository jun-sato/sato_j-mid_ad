# Download J-MID data from NII server!

SINET6に接続しているデバイスを経由してデータを取得する。

 1. 所見文から適切な画像を検索
 2. それらをserverからダウンロード
 3. SQUIDにuploadしてセグメンテーションを実行。
 4. セグメンテーション画像から使いたい臓器を抜き出す。

 **注意！SQUIDにセグメンテーションさせるときは、スライスの枚数ごとに分けたほうが効率よくセグメンテーションができる**
 

 ## 1. sshの接続~所見文から適切な画像を検索。

 SINET6接続されているパソコンのterminalから以下でログイン。(秘密鍵は~/.ssh/niiにあるという前提。)
```
ssh -i ~/.ssh/nii usropsaiE1@10.1.10.52
```
ディレクトリ/home/users/Share3/radiology/nifti_and_findings_daily/CT/nifti  に年度ごとに画像のデータファイルとnifti拡張子の画像がある。
* まずはcsvのみを全てdownload
```
scp -i ~/.ssh/nii usropsaiE1@10.1.10.52:/home/users/Share3/radiology/nifti_and_findings_daily/CT/nifti/202*/*.csv ../data
```
これでdataディレクトリに保存できる全ての画像の情報が手に入る。
次に、画像情報を利用して実際に画像を抽出する作業を行う。2022年度は構造化データが先に届かなかったので、ダウンロードしたcsvファイルを利用して抽出を行ったが、本来は構造化所見文から特定の疾患を持つものを選んできたほうがいい。
* extract_abdomen.pyを利用する。
```
## csvファイルは ../data 内に入っていることが前提。
## ここで、スライス枚数40枚以下のものはダウンロードしない。
python extract_abdomen.py
```
これにより、../data内に**include_abdomen_{開始日付}_{終了日付}.csv** と **include_abdomen_path_{開始日付}_{終了日付}.csv**というファイルが出力される。このうち、pathのついた方を実際のダウンロードのために使用する。

## 2. serverからダウンロード。
* ダウンロードファイル(**abd_download.sh**)を使ってscpダウンロード。
```
## 3行目はinclude_abdomen_pathファイルのパスを書く。
bash abd_download.sh
```
これで指定された場所にniftiファイルが保存されるはず！！(所要時間1日/1ヶ月分)

## 3. SQUIDにuploadしてセグメンテーションを実行。

まず、scpコマンドを使ってlocalからSQUID上にデータをuploadする。
```
## K22A11以下は各々のpathを指定。
scp -r path/to/the/dataset u6b588@squidhpc.hpc.cmc.osaka-u.ac.jp:/sqfs/work/K22A11/u6b588/jmid/data/
```
uploadされたデータを、renameして、スライス厚ごとに分けてディレクトリを分割する。(←効率よくセグメンテーションを実施するため。)
* rename_jmid_ver2.pyを使用。
```
python rename_jmid_ver2.py --datadir ../data --start_id 4000 --num_threads 20
## --datadir uploadしたniftiファイルのpath
## --start_id 以前にいくつかrenameを実行していた場合、番号が重なるので、以前の番号の後から始める。
## --num_threads マルチプロセス数
```
これで、niftiファイルがスライス数ごとに一定数ディレクトリに保存された。
次に、nn-UNetを用いてセグメンテーションを行う。
ディレクトリごとに.shファイルを作成しSQUID上で並列実行していく。
(e.g. nnunet_pred_5_1.sh)
予測が終了したものを集めて一つのディレクトトリ(finish_img/pred)にまとめる。
* extract_finished_file.pyを使用。
```
python extract_finished_file.py
```
予測できたファイルのリストがfinished.csvとして出力される。
これでCT画像から多臓器セグメンテーションをした予測ファイルを作成できた。これは全体では扱えないので、特定臓器のみを抽出させて後の処理に繋げる。
## 4. セグメンテーション画像から使いたい臓器を抜き出す。

* crop_dataset.pyを使用
```
python crop_dataset.py --maskdir /path/to/maskdir/ --imagedir /path/to/imagedir --save_maskdir /path/to/saved/maskdir --save_imagedir /path/to/saved/maskdir --num_threads 20 --organ_id 8
## --maskdir セグメンテーションの予測フォルダ
## --imagedir 元画像のフォルダ
## --save_maskdir 特定の臓器を切り取った後に予測を保存するフォルダ
## --save_imagedir 特定の臓器を切り取った後に元画像を保存するフォルダ
##--num_threads 20　マルチプロセス数
##--organ_id 8　臓器id(crop_dataset.py内を参照)
```