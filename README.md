# Labelme2YOLO

**Forked from [rooneysh/Labelme2YOLO](https://github.com/rooneysh/Labelme2YOLO)**

[![PyPI - Version](https://img.shields.io/pypi/v/labelme2yolo.svg)](https://pypi.org/project/labelme2yolo)
![PyPI - Downloads](https://img.shields.io/pypi/dm/labelme2yolo?style=flat)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/labelme2yolo.svg)](https://pypi.org/project/labelme2yolo)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/12122fe86f8643c4aa5667c20d528f61)](https://www.codacy.com/gh/GreatV/labelme2yolo/dashboard?utm_source=github.com\&utm_medium=referral\&utm_content=GreatV/labelme2yolo\&utm_campaign=Badge_Grade)

Labelme2YOLO is a powerful tool for converting LabelMe's JSON format to [YOLOv5](https://github.com/ultralytics/yolov5) dataset format. This tool can also be used for YOLOv5/YOLOv8 segmentation datasets, if you have already made your segmentation dataset with LabelMe, it is easy to use this tool to help convert to YOLO format dataset.

## New Features

* export data as yolo polygon annotation (for YOLOv5 v7.0 segmentation)
* Now you can choose the output format of the label text. The two available alternatives are `polygon` and bounding box (`bbox`).

## Installation

```shell
pip install labelme2yolo
```

## Arguments

**--json\_dir** LabelMe JSON files folder path.

**--val\_size (Optional)** Validation dataset size, for example 0.2 means 20% for validation.

**--test\_size (Optional)** Test dataset size, for example 0.2 means 20% for Test.

**--json\_name (Optional)** Convert single LabelMe JSON file.

**--output\_format (Optional)** The output format of label.

**--label\_list (Optional)** The pre-assigned category labels.

## How to Use

### 1. Converting JSON files and splitting training, validation, and test datasets with --val\_size and --test\_size

You may need to place all LabelMe JSON files under **labelme\_json\_dir** and then run the following command:

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

### 2. Converting JSON files and splitting training and validation datasets by folders

If you have split the LabelMe training dataset and validation dataset on your own, please put these folders under **labelme\_json\_dir** as shown below:

```plaintext
/path/to/labelme_json_dir/train/
/path/to/labelme_json_dir/val/
```

This tool will read the training and validation datasets by folder. You may run the following command to do this:

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

## How to build package/wheel

1. [install hatch](https://hatch.pypa.io/latest/install/)
2. Run the following command:

```shell
hatch build
```

## License

`labelme2yolo` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
