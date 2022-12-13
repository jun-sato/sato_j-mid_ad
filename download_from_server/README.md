# Download J-MID data from NII server!

SINET6に接続しているデバイスを経由してデータを取得する。

 1. 所見文から適切な画像を検索
 2. それらをserverからダウンロード
 3. SQUIDにuploadしてセグメンテーションを実行。
 4. セグメンテーション画像から使いたい臓器を抜き出す。

 **注意！SQUIDにセグメンテーションさせるときは、スライスの枚数ごとに分けたほうが効率よくセグメンテーションができる**
 

 ## 1. sshの接続~所見文から適切な画像を検索。

 SINET6接続されているパソコンのterminalから以下でログイン。(秘密鍵は~/.ssh/nii)にあるという前提。
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
