# CADTransformer: CAD図面のためのパノプティックシンボル検出トランスフォーマー

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

CVPR2022で口頭発表された論文の公式PyTorch実装

## プロジェクト概要

CADTransformerは、CAD図面（特に建築の平面図）内のシンボルを検出・分類するための深層学習モデルです。このモデルは、Transformerアーキテクチャを活用して、CAD図面内の様々な要素（ドア、窓、家具など）を高精度で識別します。

### 主な特徴

- **パノプティックシンボル検出**: CAD図面内の様々なシンボルを検出し、分類します
- **Transformerベースのアーキテクチャ**: 自己注意機構を活用して、シンボル間の関係性を捉えます
- **HRNetとViTの組み合わせ**: 特徴抽出にHRNetを使用し、Transformerエンコーダとして修正版のViTを使用します

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