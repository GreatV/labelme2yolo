# Labelme2YOLO

[![PyPI - Version](https://img.shields.io/pypi/v/labelme2yolo.svg)](https://pypi.org/project/labelme2yolo)
![PyPI - Downloads](https://img.shields.io/pypi/dm/labelme2yolo?style=flat)
[![PYPI - Downloads](https://static.pepy.tech/badge/labelme2yolo)](https://pepy.tech/project/labelme2yolo)

Labelme2YOLO efficiently converts LabelMe's JSON format to the YOLOv5 dataset format. It also supports YOLOv5/YOLOv8 segmentation datasets, making it simple to convert existing LabelMe segmentation datasets to YOLO format.

## New Features

* export data as yolo polygon annotation (for YOLOv5 & YOLOV8 segmentation)
* Now you can choose the output format of the label text. The two available alternatives are `polygon` and bounding box(`bbox`).

## Performance

Labelme2YOLO is implemented in Rust, which makes it significantly faster than equivalent Python implementations. In fact, it can be up to 100 times faster, allowing you to process large datasets more efficiently.

## Installation

```shell
pip install labelme2yolo
```

## Arguments

**[LABEL_LIST]...** Comma-separated list of labels in the dataset.

## Options

**-d, --json_dir <JSON_DIR>** Directory containing LabelMe JSON files.

**--val_size <VAL_SIZE>** Proportion of the dataset to use for validation (between 0.0 and 1.0) [default: 0.2].

**--test_size <TEST_SIZE>** Proportion of the dataset to use for testing (between 0.0 and 1.0) [default: 0].

**--output_format <OUTPUT_FORMAT>** Output format for YOLO annotations: 'bbox' or 'polygon' [default: bbox] [aliases: format] [possible values: polygon, bbox].

**--seed <SEED>** Seed for random shuffling [default: 42].

**-h, --help** Print help.

**-V, --version** Print version.

## How to Use

### 1. Converting JSON files and splitting training, validation datasets

You may need to place all LabelMe JSON files under **labelme_json_dir** and then run the following command:

```shell
labelme2yolo --json_dir /path/to/labelme_json_dir/
```

This tool will generate dataset labels and images with YOLO format in different folders, such as

```plaintext
/path/to/labelme_json_dir/YOLODataset/labels/train/
/path/to/labelme_json_dir/YOLODataset/labels/val/
/path/to/labelme_json_dir/YOLODataset/images/train/
/path/to/labelme_json_dir/YOLODataset/images/val/
/path/to/labelme_json_dir/YOLODataset/dataset.yaml
```

### 2. Converting JSON files and splitting training, validation, and test datasets with --val_size and --test_size

You may need to place all LabelMe JSON files under **labelme_json_dir** and then run the following command:

```shell
labelme2yolo --json_dir /path/to/labelme_json_dir/ --val_size 0.15 --test_size 0.15
```

This tool will generate dataset labels and images with YOLO format in different folders, such as

```plaintext
/path/to/labelme_json_dir/YOLODataset/labels/train/
/path/to/labelme_json_dir/YOLODataset/labels/test/
/path/to/labelme_json_dir/YOLODataset/labels/val/
/path/to/labelme_json_dir/YOLODataset/images/train/
/path/to/labelme_json_dir/YOLODataset/images/test/
/path/to/labelme_json_dir/YOLODataset/images/val/
/path/to/labelme_json_dir/YOLODataset/dataset.yaml
```

## How to build package/wheel

```shell
pip install maturin
maturin develop
```
