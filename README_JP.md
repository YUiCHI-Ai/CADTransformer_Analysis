# CADTransformer: CAD図面のためのパノプティックシンボル検出トランスフォーマー

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

CVPR2022で口頭発表された論文の公式PyTorch実装

## プロジェクト概要

CADTransformerは、CAD図面（特に建築の平面図）内のシンボルを検出・分類するための深層学習モデルです。このモデルは、Transformerアーキテクチャを活用して、CAD図面内の様々な要素（ドア、窓、家具など）を高精度で識別します。

### 主な特徴

- **パノプティックシンボル検出**: CAD図面内の様々なシンボルを検出し、分類します
- **Transformerベースのアーキテクチャ**: 自己注意機構を活用して、シンボル間の関係性を捉えます
- **HRNetとViTの組み合わせ**: 特徴抽出にHRNetを使用し、Transformerエンコーダとして修正版のViTを使用します

### 識別可能な要素

CADTransformerは、以下の35種類の要素を識別することができます：

#### ドア関連（6種類）
1. 片開きドア（single door）
2. 両開きドア（double door）
3. 引き戸（sliding door）
4. 折りたたみドア（folding door）
5. 回転ドア（revolving door）
6. シャッター（rolling door）

#### 窓関連（4種類）
7. 窓（window）
8. 出窓（bay window）
9. ブラインド窓（blind window）
10. 開口部記号（opening symbol）

#### 家具関連（14種類）
11. ソファ（sofa）
12. ベッド（bed）
13. 椅子（chair）
14. テーブル（table）
15. TVキャビネット（TV cabinet）
16. ワードローブ（Wardrobe）
17. キャビネット（cabinet）
19. シンク（sink）
22. バス（bath）
23. 浴槽（bath tub）
25. 和式トイレ（squat toilet）
26. 小便器（urinal）
27. トイレ（toilet）

#### 家電関連（3種類）
18. ガスコンロ（gas stove）
20. 冷蔵庫（refrigerator）
24. 洗濯機（washing machine）

#### 設備関連（2種類）
29. エレベーター（elevator）
30. エスカレーター（escalator）

#### その他（6種類）
21. エアコン（airconditioner）
28. 階段（stairs）
31. 列席（row chairs）
32. 駐車スペース（parking spot）
33. 壁（wall）
34. カーテンウォール（curtain wall）
35. 手すり（railing）

## インストール方法

`conda`を使用して実行環境をインストールすることを推奨します。以下の依存関係が必要です：

```
CUDA=11.1
Python=3.7.7
pytorch=1.9.0
torchvision=0.10.0
sklearn=1.0.1
pillow=8.3.1
opencv-python
matplotlib
scipy
tqdm
gdown
svgpathtools
```

このコードはpytorch>=1.5.0と互換性があります。

## 事前学習済みHRNetのダウンロード

入力埋め込みネットワークはHRNet-W48-Cに基づいています。ImageNetで事前学習されたモデルは公式の[クラウドドライブ](https://github.com/HRNet/HRNet-Image-Classification)からダウンロードできます。

```
cd CADTransformer
mkdir pretrained_models
```

ダウンロードした事前学習済みHRNetをCADTransformer/pretrained_models/に配置してください。

## 学習済みCADTransformerモデルについて

**注意**: 公式リポジトリでは、CADTransformer自体の学習済みモデルは提供されていません。モデルを使用するには、上記の手順に従って自分でモデルを訓練する必要があります。訓練後は、以下のパスに最良のモデルが保存されます：

```
/PATH/TO/PROJECT_DIR/logs/[log_dir]/best_model.pth
```

このモデルは、`--load_ckpt`オプションを使用して読み込むことができます：

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_cad_ddp.py --data_root /PATH/TO/PROJECT_DIR/data/FloorPlanCAD --pretrained_model /PATH/TO/PROJECT_DIR/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth --load_ckpt /PATH/TO/PROJECT_DIR/logs/[log_dir]/best_model.pth --test_only
```

## データ準備

変換済みデータのサンプルがいくつか提供されており、公式FloorPlanCADデータセットからダウンロードしなくてもコードを実行できます。

FloorPlanCADデータセット全体でモデルを訓練するには、まず公式の[クラウドドライブ](https://floorplancad.github.io/)からデータをダウンロードし、以下のコマンドに従ってファイルを再配置する必要があります：

### floorplancadウェブサイトからのダウンロード
```
python preprocess/download_data.py --data_save_dir /ssd1/zhiwen/datasets/svg_raw
```

### セマンティックラベリングをfloorplanCAD v1バージョンに変換し、ラスタライズされた画像を生成
```
python preprocess/svg2png.py --train_00 /ssd1/zhiwen/datasets/svg_raw/train-00 --train_01 /ssd1/zhiwen/datasets/svg_raw/train-01 --test_00 /ssd1/zhiwen/datasets/svg_raw/test-00 --svg_dir /ssd1/zhiwen/datasets/svg_processed/svg --png_dir /ssd1/zhiwen/datasets/svg_processed/png --scale 7 --cvt_color
```

### npy形式のデータを生成
```
python preprocess/preprocess_svg.py -i /ssd1/zhiwen/datasets/svg_processed/svg/train -o /ssd1/zhiwen/datasets/svg_processed/npy/train --thread_num 48
python preprocess/preprocess_svg.py -i /ssd1/zhiwen/datasets/svg_processed/svg/test -o /ssd1/zhiwen/datasets/svg_processed/npy/test --thread_num 48
python preprocess/preprocess_svg.py -i /ssd1/zhiwen/datasets/svg_processed/svg/val -o /ssd1/zhiwen/datasets/svg_processed/npy/val --thread_num 48
```

```
mkdir data
ln -s /ssd1/zhiwen/datasets/svg_processed ./data/floorplancad_v2
```

データディレクトリ構造は以下のようになります：
```
├── data
├──├── FloorPlanCAD
├──├──├── npy(スクリプトを使用して変換)
│  │  │   └── test
│  │  │   └── train   
│  │  │   └── val    
├──├──├── png(スクリプトを使用して変換)
│  │  │   └── test
│  │  │   └── train  
│  │  │   └── val  
├──├──├── svg(https://floorplancad.github.io/からダウンロード)
│  │  │   └── test
│  │  │   └── train  
│  │  │   └── val  
```

## 使用方法

必要なライブラリをインストールした後、提供されたデータサンプルを使用して直接CADTransformerを訓練できます：

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_cad_ddp.py --data_root /PATH/TO/PROJECT_DIR/data/FloorPlanCAD --pretrained_model /PATH/TO/PROJECT_DIR/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
```

複数のGPUを使用して訓練プロセスを高速化することもできます：

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_cad_ddp.py --data_root /PATH/TO/PROJECT_DIR/data/FloorPlanCAD --pretrained_model /PATH/TO/PROJECT_DIR/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
```

提供されたデータサンプルを使用して、CADTransformerのテスト/検証を直接実行することもできます：

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_cad_ddp.py --data_root /PATH/TO/PROJECT_DIR/data/FloorPlanCAD --pretrained_model /PATH/TO/PROJECT_DIR/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth --test_only
```

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train_cad_ddp.py --data_root /PATH/TO/PROJECT_DIR/data/FloorPlanCAD --pretrained_model /PATH/TO/PROJECT_DIR/pretrained_models/hrnetv2_w48_imagenet_pretrained.pth --val_only
```

以下のコマンドでパノプティック品質メトリックを取得できます：

```
python scripts/evaluate_pq.py --raw_pred_dir /PATH/TO/SAVE_DIR/IN/PREVIOUS/STEP 
--svg_pred_dir /PATH/TO/PROJECT_DIR/FloorPlanCAD/svg_pred --svg_gt_dir /PATH/TO/PROJECT_DIR/FloorPlanCAD/svg_gt --thread_num 6
```

## よくある質問

#### メモリ不足エラーが発生する場合
[A]: args.max_primを通じて最大プリミティブ数を減らしてください。24GBのGPUでは12000に設定しています。

## 謝辞

Ross Wightman、qq456cvb、Ke Sunの優れた作品[pytorch-image-models](https://github.com/rwightman/pytorch-image-models)、[Point-Transformers](https://github.com/qq456cvb/Point-Transformers)、[HRNet](https://github.com/HRNet/HRNet-Image-Classification)をオープンソースにしてくれたことに感謝します。

## 引用

研究や作業に役立った場合は、以下の論文を引用してください：

```
@inproceedings{fan2022cadtransformer,
  title={CADTransformer: Panoptic Symbol Spotting Transformer for CAD Drawings},
  author={Fan, Zhiwen and Chen, Tianlong and Wang, Peihao and Wang, Zhangyang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10986--10996},
  year={2022}
}
```

## モデルアーキテクチャ

CADTransformerは以下の主要コンポーネントで構成されています：

1. **入力埋め込みネットワーク**: HRNet-W48-Cを使用して、CAD図面の画像特徴を抽出します
2. **Transformerエンコーダ**: 修正版のViT（Vision Transformer）を使用して、シンボル間の関係性を捉えます
3. **分類ヘッド**: 抽出された特徴を使用して、各シンボルのクラスを予測します

このモデルは、CAD図面内の35種類の異なるシンボル（ドア、窓、家具、設備など）を識別できます。