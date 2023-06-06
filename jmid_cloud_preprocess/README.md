## Jmid Cloudからデータをダウンロード。

今回の実験では、 NIIクラウドでダウンロードできなかったものを、JMIDクラウドを利用してDICOMとしてダウンロードするようにした。


1. 全体のデータファイルの中から、NIIクラウドでダウンロードできたデータを除いたファイルリストを作成する。  
→``dcm_download.csv``

2. JMIDクラウドの中から、全部の検査のリストをダウンロードしてきて保存する。  
→``jmid_cloud``ディレクトリ
ディレクトリ内の構造
```
jmid_cloud
 |-----list_dir1
 |        |---file_0.csv
 |        |---file_1.csv
 |         ----
 |-----list_dir2  
 |        |---file_0.csv
 |        |---file_1.csv
```
3. 1でダウンロードすべきデータを、2でcloud内での情報を取得できた。これらをmergeしてダウンロードしたいデータをダウンロードする。
mergeするファイル  
→  jmidcloud_installlist.ipynb
これを使ってダウンロードリストを取得する。

4. jmidダウンローダーを使ってDICOMをダウンロード

5. [dcm2niix](https://github.com/rordenlab/dcm2niix)を利用してdicomファイルを.nii.gzに変える。使いかたの例は以下の通り。
```bash
/home/users/Share/bin/dcm2niix -o ${DATE_FOLDER} -f ${f_basename}_${n_slice} -m y -s y -z y ${fname}
```

 上記で${f_basename}は（施設コード＋アクセッション番号＋シリーズ番号）のファイル名で、
${n_slice}はシリーズ内の画像枚数をdicomdirからカウントしたもの、
${fname}はdicomdir内のテキスト・ファイル名でそのファイルには画像の所在がリストされています。