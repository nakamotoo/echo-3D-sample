## 生データの配置

まずは生データを配置する必要があります．
normal 群の avi ファイルを ` data/raw/normal` に配置
abnormal 群の avi ファイルを ` data/raw/abnormal` に配置

## データの resize と FileList.csv の作成

続いて，配置した生データを resize して，さらに train/valid/test の FileList を作成します．
` data/FileList_Maker.ipynb ` をの前半部分は，生データを 112x112px に resize し， `data/videos/original` に保存
後半部分は FileList.csv を作成して保存

## データオーグメンテーション

` data/data_augmentation.ipynb ` でデータオーグメンテーション を行います．
使用するライブラリ vidaug を別途インストールする必要があります.手順がちょっとややこしいのですが，コード内にインストール方法を記載しています．

## スタンフォードのR2+1Dの学習済みの重み
`wget https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt`

## 学習

` train.ipynb ` で学習を行います．
